#include "Filtering.hpp"

#include "../DebugPrintFunctions.hpp"
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/NodeDisplayedTreeData.hpp"
#include "../io/NetworkIO.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/LikelihoodVariant.hpp"
#include "../optimization/NetworkState.hpp"
#include "../optimization/Optimization.hpp"
#include "../optimization/ReticulationOptimization.hpp"
#include "ScoreImprovement.hpp"

namespace netrax {

template <typename T>
struct ScoreItem {
  T item;
  double bicScore = std::numeric_limits<double>::infinity();
};

double trim(double x, int digitsAfterComma) {
  double factor = pow(10, digitsAfterComma);
  return (double)((int)(x * factor)) / factor;
}

std::unordered_set<size_t> findPromisingNodes(AnnotatedNetwork &ann_network,
                                              std::vector<double> &nodeScore,
                                              bool silent) {
  std::unordered_set<size_t> promisingNodes;
  std::vector<ScoreItem<Node *>> scoresNodes(ann_network.network.num_nodes());
  for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
    scoresNodes[i] =
        ScoreItem<Node *>{ann_network.network.nodes_by_index[i], nodeScore[i]};
  }

  std::sort(scoresNodes.begin(), scoresNodes.end(),
            [](const ScoreItem<Node *> &lhs, const ScoreItem<Node *> &rhs) {
              return lhs.bicScore < rhs.bicScore;
            });

  size_t cutoff_pos =
      std::min(ann_network.options.prefilter_keep - 1, scoresNodes.size() - 1);

  double cutoff_bic = scoresNodes[cutoff_pos].bicScore;  // best_bic;

  for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
    if (scoresNodes[i].bicScore == std::numeric_limits<double>::infinity()) {
      break;
    }
    if (scoresNodes[i].bicScore <= cutoff_bic) {
      promisingNodes.emplace(scoresNodes[i].item->clv_index);
    }

    if (promisingNodes.size() == ann_network.options.prefilter_keep) {
      break;
    }
  }
  return promisingNodes;
}

template <typename T>
void filterCandidatesByNodes(std::vector<T> &candidates,
                             const std::unordered_set<size_t> &promisingNodes) {
  std::vector<T> newCandidates;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (promisingNodes.count(candidates[i].node_orig_idx) > 0) {
      newCandidates.emplace_back(candidates[i]);
    }
  }
  candidates = newCandidates;
}

template <typename T>
void filterCandidatesByScore(std::vector<T> &candidates,
                             std::vector<ScoreItem<T>> &scores,
                             double old_score, int n_keep, bool keep_all_better,
                             bool silent) {
  std::sort(scores.begin(), scores.end(),
            [](const ScoreItem<T> &lhs, const ScoreItem<T> &rhs) {
              return lhs.bicScore < rhs.bicScore;
            });
  int newSize = 0;
  size_t cutoff_pos = std::min(n_keep, (int)scores.size() - 1);
  double cutoff_bic = scores[cutoff_pos].bicScore;
  if (keep_all_better && cutoff_bic < old_score) {
    cutoff_bic = old_score;
  }

  for (size_t i = 0; i < std::min(scores.size(), candidates.size()); ++i) {
    if (scores[i].bicScore < cutoff_bic) {
      candidates[newSize] = scores[i].item;
      newSize++;
    }
  }
  candidates.resize(newSize);
}

void advance_progress(float progress, int barWidth) {
  // progress bar code taken from https://stackoverflow.com/a/14539953/14557921
  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)
        std::cout << "=";
      else if (i == pos)
        std::cout << ">";
      else
        std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
  }
}

template <typename T>
size_t elbowMethod(const std::vector<ScoreItem<T>> &elements,
                   int max_n_keep = std::numeric_limits<int>::max()) {
  // see
  // https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera/
  // see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
  // we need to find the point with the largest distance to the line from the
  // first to the last point; this point corresponds to our chosen cutoff value.
  int minIdx = 0;

  int lastIdx = std::min((int)elements.size(), max_n_keep) - 1;

  double maxDist = 0;
  int maxDistIdx = minIdx;

  int x1 = minIdx;
  double y1 = elements[minIdx].bicScore;
  int x2 = lastIdx;
  double y2 = elements[lastIdx].bicScore;
  for (int i = minIdx + 1; i <= lastIdx;
       ++i) {  // because the endpoints trivially have distance 0
    int x0 = i;
    double y0 = elements[i].bicScore;
    double d = std::abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1);
    if (d >= maxDist) {
      maxDist = d;
      maxDistIdx = i;
    }
  }
  return maxDistIdx + 1;
}

enum class FilterType { PREFILTER = 0, RANK = 1, CHOOSE = 2 };

void optimizeAfterMove(AnnotatedNetwork &ann_network, Move &move,
                       FilterType filterType) {
  if (filterType == FilterType::PREFILTER) {
    if (isArcInsertion(move.moveType)) {
      std::unordered_set<size_t> brlenopt_candidates;
      brlenopt_candidates.emplace(
          move.arcInsertionData.wanted_uv_pmatrix_index);
      optimizeBranchesCandidates(
          ann_network, brlenopt_candidates,
          1.0 / RAXML_BRLEN_SMOOTHINGS);  // one iteration
      optimize_reticulation(ann_network,
                            ann_network.network.num_reticulations() - 1);
      updateMoveBranchLengths(ann_network, move);
    }
  } else if (filterType == FilterType::RANK) {
    std::unordered_set<size_t> brlen_opt_candidates =
        brlenOptCandidates(ann_network, move);
    assert(!brlen_opt_candidates.empty());
    optimizeBranchesCandidates(ann_network, brlen_opt_candidates);
    optimizeReticulationProbs(ann_network);
    updateMoveBranchLengths(ann_network, move);
  } else if (filterType == FilterType::CHOOSE) {
    optimizeBranches(ann_network);
    optimizeReticulationProbs(ann_network);
    updateMoveBranchLengths(ann_network, move);
  }
}

double filterCandidates(AnnotatedNetwork &ann_network,
                        const NetworkState &oldState, NetworkState &bestState,
                        std::vector<Move> &candidates, FilterType filterType,
                        bool enforce, bool extreme_greedy, bool keep_all_better,
                        bool silent, bool print_progress) {
  if (candidates.empty()) {
    return scoreNetwork(ann_network);
  }
  assert(!(extreme_greedy && enforce));
  int n_better = 0;
  int barWidth = 70;
  double old_bic = scoreNetwork(ann_network);
  double best_bic = old_bic;
  if (enforce) {
    best_bic = std::numeric_limits<double>::infinity();
  }
  std::vector<ScoreItem<Move>> scores(candidates.size());

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (print_progress) {
      advance_progress((float)(i + 1) / candidates.size(), barWidth);
    }
    Move move(candidates[i]);
    performMove(ann_network, move);
    optimizeAfterMove(ann_network, move, filterType);
    double bicScore = scoreNetwork(ann_network);
    scores[i] = ScoreItem<Move>{candidates[i], bicScore};
    if (bicScore < old_bic) {
      n_better++;
    }
    if (bicScore < best_bic) {
      best_bic = bicScore;
      extract_network_state(ann_network, bestState);
    }
    if (extreme_greedy && (bicScore < old_bic)) {
      std::swap(candidates[0], candidates[i]);
      candidates.resize(1);
      undoMove(ann_network, move);
      apply_network_state(ann_network, oldState);
      if (print_progress && ParallelContext::master_rank() &&
          ParallelContext::master_thread()) {
        std::cout << std::endl;
      }
      return best_bic;
    }
    undoMove(ann_network, move);
    assert(checkSanity(ann_network, candidates[i]));
    apply_network_state(ann_network, oldState);
  }

  apply_network_state(ann_network, oldState);
  if (print_progress && ParallelContext::master_rank() &&
      ParallelContext::master_thread()) {
    std::cout << std::endl;
  }

  size_t oldCandidatesSize = candidates.size();
  int n_keep = ann_network.options.rank_keep;
  if (!ann_network.options.no_elbow_method) {
    n_keep = elbowMethod(scores, n_keep);
  }
  assert(n_keep > 0);
  filterCandidatesByScore(candidates, scores, old_bic, n_keep, keep_all_better,
                          silent);
  if (!silent) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << "New size after ";
      if (filterType == FilterType::PREFILTER) {
        std::cout << "prefiltering: ";
      } else if (filterType == FilterType::RANK) {
        std::cout << "ranking: ";
      } else if (filterType == FilterType::CHOOSE) {
        std::cout << "choosing: ";
      }
      std::cout << candidates.size() << " vs. " << oldCandidatesSize << "\n";
    }
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    assert(checkSanity(ann_network, candidates[i]));
  }
  return best_bic;
}

double prefilterCandidates(AnnotatedNetwork &ann_network,
                           const NetworkState &oldState,
                           NetworkState &bestState,
                           std::vector<Move> &candidates, bool extreme_greedy,
                           bool silent, bool print_progress) {
  double old_bic = scoreNetwork(ann_network);
  double best_bic = filterCandidates(ann_network, oldState, bestState, candidates,
                          FilterType::PREFILTER, false, extreme_greedy, true,
                          silent, print_progress);
  if (best_bic >= old_bic) {
      candidates.clear();
  }
  return best_bic;
}

double rankCandidates(AnnotatedNetwork &ann_network,
                      const NetworkState &oldState, NetworkState &bestState,
                      std::vector<Move> &candidates, bool enforce,
                      bool extreme_greedy, bool silent, bool print_progress) {
  if (!ann_network.options.no_prefiltering) {
    prefilterCandidates(ann_network, oldState, bestState, candidates, extreme_greedy,
                        silent, print_progress);
  }
  return filterCandidates(ann_network, oldState, bestState, candidates,
                          FilterType::RANK, enforce, extreme_greedy, false,
                          silent, print_progress);
}

double chooseCandidate(AnnotatedNetwork &ann_network,
                       const NetworkState &oldState, NetworkState &bestState,
                       std::vector<Move> &candidates, bool enforce,
                       bool extreme_greedy, bool silent, bool print_progress) {
  double old_bic = scoreNetwork(ann_network);
  if (candidates.empty()) {
    return old_bic;
  }
  rankCandidates(ann_network, oldState, bestState, candidates, enforce,
                 extreme_greedy, silent, print_progress);
  double best_bic = filterCandidates(
      ann_network, oldState, bestState, candidates, FilterType::CHOOSE, enforce,
      extreme_greedy, false, silent, print_progress);
  if (best_bic >= old_bic && !enforce) {
    candidates.clear();
  }
  return best_bic;
}

double acceptMove(AnnotatedNetwork &ann_network, Move &move,
                  const NetworkState &bestState, double *best_score,
                  BestNetworkData *bestNetworkData, bool silent) {
  assert(checkSanity(ann_network, move));
  assert(computeLoglikelihood(ann_network, 1, 1) ==
         computeLoglikelihood(ann_network, 0, 1));
  performMove(ann_network, move);
  apply_network_state(ann_network, bestState);
  assert(computeLoglikelihood(ann_network, 1, 1) ==
         computeLoglikelihood(ann_network, 0, 1));
  optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::NORMAL);
  assert(computeLoglikelihood(ann_network, 1, 1) ==
         computeLoglikelihood(ann_network, 0, 1));

  double logl = computeLoglikelihood(ann_network);
  double bic_score = bic(ann_network, logl);
  double aic_score = aic(ann_network, logl);
  double aicc_score = aicc(ann_network, logl);

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    /*if (!silent) */ std::cout << " Took " << toString(move.moveType) << "\n";
    if (!silent)
      std::cout << "  Logl: " << logl << ", BIC: " << bic_score
                << ", AIC: " << aic_score << ", AICc: " << aicc_score << "\n";
    if (!silent)
      std::cout << "  param_count: " << get_param_count(ann_network)
                << ", sample_size:" << get_sample_size(ann_network) << "\n";
    if (!silent)
      std::cout << "  num_reticulations: "
                << ann_network.network.num_reticulations() << "\n";
    // if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";

    std::cout << "displayed trees:\n";
    for (size_t i = 0;
         i <
         ann_network
             .pernode_displayed_tree_data[ann_network.network.root->clv_index]
             .num_active_displayed_trees;
         ++i) {
      DisplayedTreeData &dtd =
          ann_network
              .pernode_displayed_tree_data[ann_network.network.root->clv_index]
              .displayed_trees[i];
      std::cout << "tree " << i << ":\n";
      std::cout << "  prob: " << exp(dtd.treeLoglData.tree_logprob) << "\n";
      double tree_logl =
          std::accumulate(dtd.treeLoglData.tree_partition_logl.begin(),
                          dtd.treeLoglData.tree_partition_logl.end(), 0.0);
      std::cout << "  logl: " << tree_logl << "\n";
      printReticulationChoices(dtd.treeLoglData.reticulationChoices);
    }
  }
  ann_network.stats.moves_taken[move.moveType]++;

  check_score_improvement(ann_network, best_score, bestNetworkData);
  return scoreNetwork(ann_network);
}

Move applyBestCandidate(AnnotatedNetwork &ann_network,
                        std::vector<Move> candidates, double *best_score,
                        BestNetworkData *bestNetworkData, bool enforce,
                        bool extreme_greedy, bool silent, bool print_progress) {
  double old_score = scoreNetwork(ann_network);

  NetworkState oldState = extract_network_state(ann_network);
  NetworkState bestState = extract_network_state(ann_network);
  double best_bic =
      chooseCandidate(ann_network, oldState, bestState, candidates, enforce,
                      extreme_greedy, silent, print_progress);
  assert(scoreNetwork(ann_network) == old_score);

  if (!candidates.empty()) {
    acceptMove(ann_network, candidates[0], bestState, best_score,
               bestNetworkData, silent);

    if (!enforce) {
      if (scoreNetwork(ann_network) > old_score) {
        if (ParallelContext::master_thread() &&
            ParallelContext::master_rank()) {
          std::cout << old_score << "\n";
          std::cout << scoreNetwork(ann_network) << "\n";
        }
        throw std::runtime_error(
            "Something went wrong in the network search. Suddenly, BIC is "
            "worse!");
      }
    }
    return candidates[0];
  }

  return {};
}

}  // namespace netrax