#include "CandidateSelection.hpp"
#include "../helper/Helper.hpp"

#include "../DebugPrintFunctions.hpp"
#include "../NetraxOptions.hpp"
#include "../colormod.h"  // namespace Color
#include "../graph/NodeDisplayedTreeData.hpp"
#include "../io/NetworkIO.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/PseudoLoglikelihood.hpp"
#include "../moves/Move.hpp"
#include "../moves/RNNI.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/NetworkState.hpp"
#include "../optimization/Optimization.hpp"

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

void switchLikelihoodVariant(AnnotatedNetwork &ann_network,
                             LikelihoodVariant newVariant) {
  return;

  if (ann_network.options.likelihood_variant == newVariant) {
    return;
  }
  ann_network.options.likelihood_variant = newVariant;
  ann_network.cached_logl_valid = false;
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
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (!silent)
          std::cout << "prefiltered node " << scoresNodes[i].item->clv_index
                    << " has best BIC: " << scoresNodes[i].bicScore << "\n";
      }
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
                             double old_score, int n_keep, bool keep_equal,
                             bool keep_all_better, bool silent) {
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
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      if (!silent)
        std::cout << "candidate " << i + 1 << "/" << candidates.size()
                  << " has BIC: " << scores[i].bicScore << "\n";
    }
    if (scores[i].bicScore < cutoff_bic) {
      candidates[newSize] = scores[i].item;
      newSize++;
    }
    /*if (!keep_equal && newSize == n_keep) {
    break;
}*/
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

double performMovePrefilter(AnnotatedNetwork &ann_network, Move &move,
                            LikelihoodVariant old_variant) {
  assert(checkSanity(ann_network, move));
  performMove(ann_network, move);
  assert(computeLoglikelihood(ann_network, 1, 1) ==
         computeLoglikelihood(ann_network, 0, 1));
  if (move.moveType == MoveType::ArcInsertionMove ||
      move.moveType == MoveType::DeltaPlusMove) {
    switchLikelihoodVariant(ann_network, old_variant);
    optimizeReticulationProbs(ann_network);
    std::unordered_set<size_t> brlenopt_candidates;
    brlenopt_candidates.emplace(move.arcInsertionData.wanted_uv_pmatrix_index);
    optimizeBranchesCandidates(
        ann_network, brlenopt_candidates,
        2.0 / RAXML_BRLEN_SMOOTHINGS);  // two iterations max
    updateMoveBranchLengths(ann_network, move);
    switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);
  }
  for (size_t j = 0; j < ann_network.network.num_nodes(); ++j) {
    assert(ann_network.network.nodes_by_index[j]->clv_index == j);
  }
  return scoreNetwork(ann_network);
}

double prefilterCandidates(AnnotatedNetwork &ann_network,
                           std::vector<Move> &candidates, bool silent = true,
                           bool print_progress = true,
                           bool need_best_bic = false) {
  if (candidates.empty()) {
    return scoreNetwork(ann_network);
  }

  int n_better = 0;
  std::vector<ScoreItem<Move>> scores(candidates.size());
  std::vector<double> nodeScore(ann_network.network.num_nodes(),
                                std::numeric_limits<double>::infinity());
  LikelihoodVariant old_variant = ann_network.options.likelihood_variant;
  switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);
  int barWidth = 70;
  double old_bic = scoreNetwork(ann_network);
  switchLikelihoodVariant(ann_network, old_variant);
  double real_old_bic = scoreNetwork(ann_network);
  switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);
  double best_bic = old_bic;  // std::numeric_limits<double>::infinity();
  double best_real_bic = real_old_bic;

  NetworkState oldState = extract_network_state(ann_network);

  assert(computeLoglikelihood(ann_network) ==
         computeLoglikelihood(ann_network, 0, 1));

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (print_progress) {
      advance_progress((float)(i + 1) / candidates.size(), barWidth);
    }
    Move move(candidates[i]);
    double bicScore = performMovePrefilter(ann_network, move, old_variant);
    nodeScore[move.node_orig_idx] =
        std::min(nodeScore[move.node_orig_idx], bicScore);
    scores[i] = ScoreItem<Move>{candidates[i], bicScore};

    if (bicScore < old_bic) {
      n_better++;
    }
    if (bicScore < best_bic) {
      best_bic = bicScore;
      if (need_best_bic && ann_network.network.num_reticulations() > 0 &&
          ann_network.options.likelihood_variant ==
              LikelihoodVariant::SARAH_PSEUDO) {
        switchLikelihoodVariant(ann_network, old_variant);
        double actRealBIC = scoreNetwork(ann_network);
        if (actRealBIC < best_real_bic) {
          best_real_bic = actRealBIC;
          ann_network.last_accepted_move_edge_orig_idx = move.edge_orig_idx;
          switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);
        }
        if (actRealBIC >= real_old_bic) {
          n_better--;
        }
      } else {
        best_real_bic = best_bic;
      }
    }

    double extra_offset = (ann_network.options.likelihood_variant ==
                           LikelihoodVariant::SARAH_PSEUDO)
                              ? 0.01
                              : 0.0;  // +0.01 because we tend to over-estimate
                                      // with pseudologlikelihood
    if (old_bic / bicScore > ann_network.options.greedy_factor + extra_offset) {
      switchLikelihoodVariant(ann_network, old_variant);
      optimizeReticulationProbs(ann_network);
      double real_bicScore = scoreNetwork(ann_network);
      if (real_old_bic / real_bicScore > ann_network.options.greedy_factor) {
        candidates[0] = candidates[i];
        candidates.resize(1);
        undoMove(ann_network, move);
        assert(computeLoglikelihood(ann_network) ==
               computeLoglikelihood(ann_network, 0, 1));
        apply_network_state(ann_network, oldState);
        if (print_progress && ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << std::endl;
        }
        switchLikelihoodVariant(ann_network, old_variant);
        return real_bicScore;
      } else {
        switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);
      }
    }
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));
    undoMove(ann_network, move);
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));
    assert(checkSanity(ann_network, candidates[i]));
    assert(computeLoglikelihood(ann_network) ==
           computeLoglikelihood(ann_network, 0, 1));
    apply_network_state(ann_network, oldState);

    if (old_bic != scoreNetwork(ann_network)) {
      if (ParallelContext::master_thread() && ParallelContext::master_rank()) {
        std::cout << "old_bic: " << old_bic << "\n";
        std::cout << "scoreNetwork: " << scoreNetwork(ann_network) << "\n";
      }
      throw std::runtime_error(
          "BIC after undoing move is not the same as BIC we had before");
    }
    assert(old_bic == scoreNetwork(ann_network));

    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));

    if (n_better >= ann_network.options.max_better_candidates) {
      scores.resize(n_better);
      break;
    }
  }
  assert(computeLoglikelihood(ann_network) ==
         computeLoglikelihood(ann_network, 0, 1));
  apply_network_state(ann_network, oldState);

  if (print_progress && ParallelContext::master_rank() &&
      ParallelContext::master_thread()) {
    std::cout << std::endl;
  }

  size_t oldCandidatesSize = candidates.size();
  // std::unordered_set<size_t> promisingNodes = findPromisingNodes(ann_network,
  // nodeScore, silent); filterCandidatesByNodes(candidates, promisingNodes);

  int n_keep = ann_network.options.prefilter_keep;
  if (!ann_network.options.no_elbow_method) {
    n_keep = elbowMethod(scores, n_keep);
  }

  filterCandidatesByScore(candidates, scores, old_bic, n_keep, false, true,
                          silent);

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    if (print_progress)
      std::cout << "New size candidates after prefiltering: "
                << candidates.size() << " vs. " << oldCandidatesSize << "\n";
  }

  for (size_t i = 0; i < candidates.size(); ++i) {
    assert(checkSanity(ann_network, candidates[i]));
  }
  switchLikelihoodVariant(ann_network, old_variant);

  return best_real_bic;
}

void rankCandidates(AnnotatedNetwork &ann_network,
                    std::vector<Move> &candidates, bool silent = true,
                    bool print_progress = true) {
  if (candidates.empty()) {
    return;
  }
  if (!ann_network.options.no_prefiltering) {
    prefilterCandidates(ann_network, candidates, silent, print_progress);
  }

  if (ann_network.options.rank_keep >= candidates.size()) {
    return;  // we would keep all anyway...
  }

  int n_better = 0;
  int barWidth = 70;
  double old_bic = scoreNetwork(ann_network);
  double best_bic = std::numeric_limits<double>::infinity();
  NetworkState oldState = extract_network_state(ann_network);
  std::vector<ScoreItem<Move>> scores(candidates.size());

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (print_progress) {
      advance_progress((float)(i + 1) / candidates.size(), barWidth);
    }

    Move move(candidates[i]);

    if (!checkSanity(ann_network, move)) {
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "problematic candidate " << i << ": " << toString(move)
                  << "\n";
      }
    }
    assert(checkSanity(ann_network, move));

    performMove(ann_network, move);
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));

    std::unordered_set<size_t> brlen_opt_candidates =
        brlenOptCandidates(ann_network, move);
    assert(!brlen_opt_candidates.empty());
    // add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
    optimizeBranchesCandidates(ann_network, brlen_opt_candidates);
    optimizeReticulationProbs(ann_network);
    updateMoveBranchLengths(ann_network, move);

    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));
    double bicScore = scoreNetwork(ann_network);

    scores[i] = ScoreItem<Move>{candidates[i], bicScore};

    for (size_t j = 0; j < ann_network.network.num_nodes(); ++j) {
      assert(ann_network.network.nodes_by_index[j]->clv_index == j);
    }

    if (bicScore < best_bic) {
      best_bic = bicScore;
      ann_network.last_accepted_move_edge_orig_idx = move.edge_orig_idx;
    }

    if (bicScore < old_bic) {
      n_better++;
    }

    if (old_bic / bicScore > ann_network.options.greedy_factor) {
      candidates[0] = candidates[i];
      candidates.resize(1);
      undoMove(ann_network, move);
      apply_network_state(ann_network, oldState);
      if (print_progress && ParallelContext::master_rank() &&
          ParallelContext::master_thread()) {
        std::cout << std::endl;
      }
      return;
    }

    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));
    undoMove(ann_network, move);
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));
    assert(checkSanity(ann_network, candidates[i]));
    apply_network_state(ann_network, oldState);
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));

    if (n_better >= ann_network.options.max_better_candidates) {
      scores.resize(n_better);
      break;
    }
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
  filterCandidatesByScore(candidates, scores, old_bic, n_keep, false, false,
                          silent);
  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    if (print_progress)
      std::cout << "New size after ranking: " << candidates.size() << " vs. "
                << oldCandidatesSize << "\n";
  }

  for (size_t i = 0; i < candidates.size(); ++i) {
    assert(checkSanity(ann_network, candidates[i]));
  }
}

double chooseCandidate(AnnotatedNetwork &ann_network,
                       std::vector<Move> &candidates, NetworkState *state,
                       bool enforce, bool silent = true,
                       bool print_progress = true) {
  double old_bic = scoreNetwork(ann_network);
  double best_bic = old_bic;
  if (enforce) {
    best_bic = std::numeric_limits<double>::infinity();
  }

  if (candidates.empty()) {
    return best_bic;
  }
  rankCandidates(ann_network, candidates, silent, print_progress);

  int n_better = 0;
  int barWidth = 70;
  NetworkState oldState = extract_network_state(ann_network);
  std::vector<ScoreItem<Move>> scores(candidates.size());
  assert(computeLoglikelihood(ann_network, 1, 1) ==
         computeLoglikelihood(ann_network, 0, 1));

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (print_progress) {
      advance_progress((float)(i + 1) / candidates.size(), barWidth);
    }
    Move move(candidates[i]);

    assert(checkSanity(ann_network, move));

    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));
    performMove(ann_network, move);
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));
    optimizeReticulationProbs(ann_network);

    optimizeBranches(ann_network);
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));

    double bicScore = scoreNetwork(ann_network);

    if (bicScore < best_bic) {
      best_bic = bicScore;
      ann_network.last_accepted_move_edge_orig_idx = move.edge_orig_idx;
      extract_network_state(ann_network, *state);
    }

    if (bicScore < old_bic) {
      n_better++;
    }

    if (old_bic / bicScore > ann_network.options.greedy_factor) {
      candidates[0] = candidates[i];
      candidates.resize(1);
      undoMove(ann_network, move);
      apply_network_state(ann_network, oldState);
      if (print_progress && ParallelContext::master_rank() &&
          ParallelContext::master_thread()) {
        std::cout << std::endl;
      }
      return best_bic;
    }

    scores[i] = ScoreItem<Move>{candidates[i], bicScore};

    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));

    undoMove(ann_network, move);
    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));

    assert(checkSanity(ann_network, candidates[i]));

    apply_network_state(ann_network, oldState);

    assert(computeLoglikelihood(ann_network, 1, 1) ==
           computeLoglikelihood(ann_network, 0, 1));

    if (n_better >= ann_network.options.max_better_candidates) {
      scores.resize(n_better);
      break;
    }
  }
  if (print_progress && ParallelContext::master_rank() &&
      ParallelContext::master_thread()) {
    std::cout << std::endl;
  }
  apply_network_state(ann_network, oldState);
  assert(scoreNetwork(ann_network) == old_bic);
  filterCandidatesByScore(candidates, scores, old_bic, candidates.size(), false,
                          false, silent);

  return best_bic;
}

double acceptMove(AnnotatedNetwork &ann_network, Move &move,
                  double expected_bic, const NetworkState &state,
                  double *best_score, BestNetworkData *bestNetworkData,
                  bool silent = true) {
  assert(checkSanity(ann_network, move));

  performMove(ann_network, move);
  apply_network_state(ann_network, state);

  double newScore = scoreNetwork(ann_network);
  if (newScore != expected_bic) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << "expected_bic: " << expected_bic << "\n";
      std::cout << "newScore: " << newScore << "\n";
    }
    throw std::runtime_error("These scores should be the same");
  }
  assert(newScore == expected_bic);

  optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::QUICK);

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
    if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";

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
                        bool silent) {
  double old_score = scoreNetwork(ann_network);

  NetworkState state = extract_network_state(ann_network);
  double best_bic =
      chooseCandidate(ann_network, candidates, &state, enforce, silent);

  bool found_better_state =
      (enforce ? (best_bic != std::numeric_limits<double>::infinity())
               : (best_bic < old_score));

  assert(scoreNetwork(ann_network) == old_score);

  if (found_better_state) {
    Move move(candidates[0]);
    acceptMove(ann_network, move, best_bic, state, best_score, bestNetworkData,
               silent);

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
    return move;
  }

  return {};
}

double best_fast_improvement(AnnotatedNetwork &ann_network, MoveType type,
                             const std::vector<MoveType> &typesBySpeed,
                             bool silent, int min_radius, int max_radius) {
  bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                                  MoveType::RSPR1Move) != typesBySpeed.end());
  bool delta_plus_present =
      (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                 MoveType::DeltaPlusMove) != typesBySpeed.end());
  double score = scoreNetwork(ann_network);

  std::vector<Move> candidates =
      possibleMoves(ann_network, type, rspr1_present, delta_plus_present,
                    min_radius, max_radius);
  score = prefilterCandidates(ann_network, candidates, silent, true, true);

  return score;
}

int findBestMaxDistance(AnnotatedNetwork &ann_network, MoveType type,
                        const std::vector<MoveType> &typesBySpeed,
                        int step_size, bool silent) {
  int best_max_distance = -1;
  if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove ||
      type == MoveType::DeltaMinusMove) {
    best_max_distance = ann_network.options.max_rearrangement_distance;
  } else {
    double old_score = scoreNetwork(ann_network);
    int act_max_distance = 0;
    int old_max_distance = 0;
    double old_greedy_factor = ann_network.options.greedy_factor;
    ann_network.options.greedy_factor = 1.0;
    NetworkState oldState = extract_network_state(ann_network);
    while (act_max_distance < ann_network.options.max_rearrangement_distance) {
      act_max_distance =
          std::min(act_max_distance + step_size,
                   ann_network.options.max_rearrangement_distance);
      double score =
          best_fast_improvement(ann_network, type, typesBySpeed, silent,
                                old_max_distance, act_max_distance);
      if (score < old_score) {
        old_max_distance = act_max_distance + 1;
        best_max_distance = act_max_distance;
      } else {
        assert(score == old_score);
        break;
      }
      apply_network_state(ann_network, oldState);
    }
    ann_network.options.greedy_factor = old_greedy_factor;
  }
  return best_max_distance;
}

void recollectFirstParents(Network &network, std::vector<Move> &candidates) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    Move &move = candidates[i];
    if (move.moveType == MoveType::RNNIMove) {
      Node *u = network.nodes_by_index[move.rnniData.u_clv_index];
      Node *v = network.nodes_by_index[move.rnniData.v_clv_index];
      Node *s = network.nodes_by_index[move.rnniData.s_clv_index];
      Node *t = network.nodes_by_index[move.rnniData.t_clv_index];

      if (u && u->getType() == NodeType::RETICULATION_NODE) {
        move.rnniData.u_first_parent_clv_index =
            getReticulationFirstParent(network, u)->clv_index;
      }
      if (v && v->getType() == NodeType::RETICULATION_NODE) {
        move.rnniData.v_first_parent_clv_index =
            getReticulationFirstParent(network, v)->clv_index;
      }
      if (s && s->getType() == NodeType::RETICULATION_NODE) {
        move.rnniData.s_first_parent_clv_index =
            getReticulationFirstParent(network, s)->clv_index;
      }
      if (t && t->getType() == NodeType::RETICULATION_NODE) {
        move.rnniData.t_first_parent_clv_index =
            getReticulationFirstParent(network, t)->clv_index;
      }
    } else if (isArcInsertion(move.moveType)) {
      Node *b = network.nodes_by_index[move.arcInsertionData.b_clv_index];
      Node *d = network.nodes_by_index[move.arcInsertionData.d_clv_index];
      if (b && b->getType() == NodeType::RETICULATION_NODE) {
        move.arcInsertionData.b_first_parent_clv_index =
            getReticulationFirstParent(network, b)->clv_index;
      }
      if (d && d->getType() == NodeType::RETICULATION_NODE) {
        move.arcInsertionData.d_first_parent_clv_index =
            getReticulationFirstParent(network, d)->clv_index;
      }
    } else if (isArcRemoval(move.moveType)) {
      Node *v = network.nodes_by_index[move.arcRemovalData.v_clv_index];
      if (v && v->getType() == NodeType::RETICULATION_NODE) {
        move.arcRemovalData.v_first_parent_clv_index =
            getReticulationFirstParent(network, v)->clv_index;
      }
    } else if (isRSPR(move.moveType)) {
      Node *y_prime = network.nodes_by_index[move.rsprData.y_prime_clv_index];
      Node *y = network.nodes_by_index[move.rsprData.y_clv_index];
      Node *z = network.nodes_by_index[move.rsprData.z_clv_index];
      if (y_prime && y_prime->getType() == NodeType::RETICULATION_NODE) {
        move.rsprData.y_prime_first_parent_clv_index =
            getReticulationFirstParent(network, y_prime)->clv_index;
      }
      if (y && y->getType() == NodeType::RETICULATION_NODE) {
        move.rsprData.y_first_parent_clv_index =
            getReticulationFirstParent(network, y)->clv_index;
      }
      if (z && z->getType() == NodeType::RETICULATION_NODE) {
        move.rsprData.z_first_parent_clv_index =
            getReticulationFirstParent(network, z)->clv_index;
      }
    }
  }
}

void updateOldCandidates(AnnotatedNetwork &ann_network, const Move &chosenMove,
                         std::vector<Move> &candidates) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    for (size_t j = 0; j < chosenMove.remapped_clv_indices.size(); ++j) {
      updateMoveClvIndex(candidates[i],
                         chosenMove.remapped_clv_indices[j].first,
                         chosenMove.remapped_clv_indices[j].second, true);
    }
    for (size_t j = 0; j < chosenMove.remapped_pmatrix_indices.size(); ++j) {
      updateMovePmatrixIndex(
          candidates[i], chosenMove.remapped_pmatrix_indices[j].first,
          chosenMove.remapped_pmatrix_indices[j].second, true);
    }
  }

  if (!candidates.empty() && candidates[0].moveType == MoveType::RNNIMove) {
    std::vector<Move> newCandidates;
    for (size_t i = 0; i < candidates.size(); ++i) {
      Move &move = candidates[i];
      Node *u = ann_network.network.nodes_by_index[move.rnniData.u_clv_index];
      Node *v = ann_network.network.nodes_by_index[move.rnniData.v_clv_index];
      Node *s = ann_network.network.nodes_by_index[move.rnniData.s_clv_index];
      Node *t = ann_network.network.nodes_by_index[move.rnniData.t_clv_index];
      std::vector<RNNIMoveType> validTypes =
          validMoveTypes(ann_network, u, v, s, t);
      for (size_t j = 0; j < validTypes.size(); ++j) {
        Move newMove(move);
        newMove.rnniData.type = validTypes[j];
        newCandidates.emplace_back(newMove);
      }
    }
    filterOutDuplicateMovesRNNI(newCandidates);
    candidates = newCandidates;
  }
  recollectFirstParents(ann_network.network, candidates);
}

std::vector<Move> fastIterationsMode(AnnotatedNetwork &ann_network,
                                     int best_max_distance, MoveType type,
                                     const std::vector<MoveType> &typesBySpeed,
                                     double *best_score,
                                     BestNetworkData *bestNetworkData,
                                     bool silent) {
  std::vector<Move> acceptedMoves;

  assert(best_max_distance >= 0);
  double old_score = scoreNetwork(ann_network);

  bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                                  MoveType::RSPR1Move) != typesBySpeed.end());
  bool delta_plus_present =
      (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                 MoveType::DeltaPlusMove) != typesBySpeed.end());
  std::vector<Move> candidates =
      possibleMoves(ann_network, type, rspr1_present, delta_plus_present, 0,
                    best_max_distance);
  prefilterCandidates(ann_network, candidates, silent);

  bool old_no_prefiltering = ann_network.options.no_prefiltering;
  ann_network.options.no_prefiltering = true;

  std::vector<Move> oldCandidates;

  bool tried_with_allnew = false;

  bool got_better = true;
  while (got_better) {
    got_better = false;
    Move chosenMove = applyBestCandidate(ann_network, candidates, best_score,
                                         bestNetworkData, false, silent);
    if (chosenMove.moveType != MoveType::INVALID) {
      acceptedMoves.emplace_back(chosenMove);
    }
    double score = scoreNetwork(ann_network);
    check_score_improvement(ann_network, best_score, bestNetworkData);
    if (score < old_score) {
      if (chosenMove.moveType != MoveType::INVALID) {
        tried_with_allnew = false;
      }
      got_better = true;
      old_score = score;

      bool hadBadReticulationAfterInsertingArc = false;
      std::vector<Move> takenRemovals;

      if (/*hasBadReticulation(ann_network) && */ isArcInsertion(
          type)) {  // try doing arc removal moves
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout
              << /*"Bad reticulation detected. */ "Trying arc removal moves.\n";
        }
        takenRemovals = fastIterationsMode(
            ann_network, ann_network.options.max_rearrangement_distance,
            MoveType::ArcRemovalMove, typesBySpeed, best_score, bestNetworkData,
            silent);
        acceptedMoves.insert(acceptedMoves.end(), takenRemovals.begin(),
                             takenRemovals.end());
        optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::NORMAL);
        check_score_improvement(ann_network, best_score, bestNetworkData);
        old_score = scoreNetwork(ann_network);
        hadBadReticulationAfterInsertingArc = (!takenRemovals.empty());
      }

      if (chosenMove.moveType != MoveType::INVALID) {
        if (std::find(oldCandidates.begin(), oldCandidates.end(), chosenMove) !=
            oldCandidates.end()) {
          if (ParallelContext::master_rank() &&
              ParallelContext::master_thread()) {
            std::cout << " Info: Taken move was in the old candidates.\n";
          }
        }
      }

      // add new possible moves to the candidate list
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "We have " << candidates.size()
                  << " candidates before removing the old bad ones.\n";
      }
      if (!hadBadReticulationAfterInsertingArc) {
        if (chosenMove.moveType != MoveType::INVALID) {
          updateOldCandidates(ann_network, chosenMove, candidates);
        }
      } else {
        for (size_t i = 0; i < takenRemovals.size(); ++i) {
          updateOldCandidates(ann_network, takenRemovals[i], candidates);
        }
      }
      removeBadCandidates(ann_network, candidates);

      oldCandidates = candidates;

      if (!hadBadReticulationAfterInsertingArc) {
        std::vector<Node *> start_nodes =
            gatherStartNodes(ann_network, chosenMove);
        std::vector<Move> moreMoves =
            possibleMoves(ann_network, type, start_nodes, rspr1_present,
                          delta_plus_present, 0, best_max_distance);
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << "Adding " << moreMoves.size() << " candidates to the "
                    << candidates.size() << " previous ones.\n ";
        }
        candidates.insert(std::end(candidates), std::begin(moreMoves),
                          std::end(moreMoves));
      }

      if (candidates.empty()) {  // no old candidates to reuse. Thus,
        // completely gather new ones.
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << "no old candidates to reuse. Thus, completely gather "
                       "new ones.\n";
        }
        tried_with_allnew = true;
        candidates = possibleMoves(ann_network, type, rspr1_present,
                                   delta_plus_present, 0, best_max_distance);
        oldCandidates.clear();
      }
      prefilterCandidates(ann_network, candidates, silent);
    } else {  // score did not get better
      if (!tried_with_allnew && !acceptedMoves.empty()) {
        tried_with_allnew = true;
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << "retrying with all new candidates\n";
        }
        candidates = possibleMoves(ann_network, type, rspr1_present,
                                   delta_plus_present, 0, best_max_distance);
        oldCandidates.clear();
        got_better = true;
      }
    }
  }

  ann_network.options.no_prefiltering = old_no_prefiltering;
  return acceptedMoves;
}

double slowIterationsMode(AnnotatedNetwork &ann_network, MoveType type,
                          int step_size,
                          const std::vector<MoveType> &typesBySpeed,
                          double *best_score, BestNetworkData *bestNetworkData,
                          bool silent) {
  double old_score = scoreNetwork(ann_network);
  check_score_improvement(ann_network, best_score, bestNetworkData);

  bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                                  MoveType::RSPR1Move) != typesBySpeed.end());
  bool delta_plus_present =
      (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                 MoveType::DeltaPlusMove) != typesBySpeed.end());

  bool old_no_prefiltering = ann_network.options.no_prefiltering;
  ann_network.options.no_prefiltering = true;

  int min_dist = 0;
  int max_dist = step_size;
  if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove ||
      type == MoveType::DeltaMinusMove) {
    max_dist = ann_network.options.max_rearrangement_distance;
  }
  bool got_better = true;
  while (got_better) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << " current distance range: [" << min_dist << "," << max_dist
                << "]\n";
    }
    got_better = false;
    std::vector<Move> candidates =
        possibleMoves(ann_network, type, rspr1_present, delta_plus_present,
                      min_dist, max_dist);
    applyBestCandidate(ann_network, candidates, best_score, bestNetworkData,
                       false, silent);
    double score = scoreNetwork(ann_network);
    if (score < old_score) {
      got_better = true;
      old_score = score;
      min_dist = 0;
      max_dist = step_size;
      if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove ||
          type == MoveType::DeltaMinusMove) {
        max_dist = ann_network.options.max_rearrangement_distance;
      }
    } else if (max_dist < ann_network.options.max_rearrangement_distance) {
      got_better = true;
      min_dist = max_dist + 1;
      max_dist = std::min(max_dist + step_size,
                          ann_network.options.max_rearrangement_distance);
    }
  }
  ann_network.options.no_prefiltering = old_no_prefiltering;
  return scoreNetwork(ann_network);
}

double fullSearch(AnnotatedNetwork &ann_network, MoveType type,
                  const std::vector<MoveType> &typesBySpeed, double *best_score,
                  BestNetworkData *bestNetworkData, bool silent) {
  double old_score = scoreNetwork(ann_network);

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    Color::Modifier blue(Color::FG_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
    std::cout << blue;
    std::cout << "\nStarting full search for move type: " << toString(type)
              << "\n";
    std::cout << def;
  }

  int step_size = 5;

  // step 1: find best max distance
  int best_max_distance;
  if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove ||
      type == MoveType::DeltaMinusMove) {
    best_max_distance = ann_network.options.max_rearrangement_distance;
  } else {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << toString(type) << " step1: find best max distance\n";
    }
    best_max_distance =
        findBestMaxDistance(ann_network, type, typesBySpeed, step_size, silent);
  }

  // step 2: fast iterations mode, with the best max distance
  if (best_max_distance >= 0) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << "\n"
                << toString(type)
                << " step 2: fast iterations mode, with the best max distance "
                << best_max_distance << "\n";
    }
    fastIterationsMode(ann_network, best_max_distance, type, typesBySpeed,
                       best_score, bestNetworkData, silent);
    optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::NORMAL);
    check_score_improvement(ann_network, best_score, bestNetworkData);
  }

  // step 3: slow iterations mode, with increasing max distance
  if (ann_network.options.slow_mode && type != MoveType::ArcRemovalMove &&
      type != MoveType::DeltaPlusMove) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout
          << "\n"
          << toString(type)
          << " step 3: slow iterations mode, with increasing max distance\n";
    }
    optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::QUICK);
    slowIterationsMode(ann_network, type, step_size, typesBySpeed, best_score,
                       bestNetworkData, silent);
  }

  optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::NORMAL);
  old_score = scoreNetwork(ann_network);
  check_score_improvement(ann_network, best_score, bestNetworkData);

  return old_score;
}

}  // namespace netrax
