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
#include "../optimization/ReticulationOptimization.hpp"
#include "Filtering.hpp"
#include "PromisingState.hpp"

namespace netrax {

std::vector<Move> getPossibleMoves(AnnotatedNetwork &ann_network,
                                   const std::vector<MoveType> &typesBySpeed,
                                   MoveType type, int min_radius,
                                   int max_radius) {
  bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                                  MoveType::RSPR1Move) != typesBySpeed.end());
  bool delta_plus_present =
      (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                 MoveType::DeltaPlusMove) != typesBySpeed.end());
  std::vector<Move> candidates =
      possibleMoves(ann_network, type, rspr1_present, delta_plus_present,
                    min_radius, max_radius);
  return candidates;
}

double best_fast_improvement(AnnotatedNetwork &ann_network,
                             PromisingStateQueue &psq,
                             const NetworkState &oldState,
                             NetworkState &bestState, MoveType type,
                             const std::vector<MoveType> &typesBySpeed,
                             int min_radius, int max_radius,
                             bool print_progress) {
  std::vector<Move> candidates =
      getPossibleMoves(ann_network, typesBySpeed, type, min_radius, max_radius);
  return prefilterCandidates(ann_network, psq, oldState,
                             scoreNetwork(ann_network), bestState, candidates,
                             true, true, print_progress);
}

int findBestMaxDistance(AnnotatedNetwork &ann_network, PromisingStateQueue &psq,
                        MoveType type,
                        const std::vector<MoveType> &typesBySpeed,
                        int step_size, bool print_progress) {
  int best_max_distance = -1;
  if (type == MoveType::RNNIMove || isArcRemoval(type)) {
    best_max_distance = ann_network.options.max_rearrangement_distance;
  } else {
    double old_score = scoreNetwork(ann_network);
    int act_max_distance = 0;
    int old_max_distance = 0;
    NetworkState oldState = extract_network_state(ann_network);
    NetworkState bestState = extract_network_state(ann_network);
    while (act_max_distance < ann_network.options.max_rearrangement_distance) {
      act_max_distance =
          std::min(act_max_distance + step_size,
                   ann_network.options.max_rearrangement_distance);
      double score = best_fast_improvement(
          ann_network, psq, oldState, bestState, type, typesBySpeed,
          old_max_distance, act_max_distance, print_progress);
      if (score < old_score) {
        old_max_distance = act_max_distance + 1;
        best_max_distance = act_max_distance;
        if (isArcInsertion(type)) {
          apply_network_state(ann_network, oldState);
          break;
        }
      } else {
        assert(score == old_score);
        break;
      }
      apply_network_state(ann_network, oldState);
    }
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

void updateCandidateMoves(AnnotatedNetwork &ann_network,
                          const std::vector<MoveType> &typesBySpeed,
                          int best_max_distance, const Move &chosenMove,
                          const std::vector<Move> &takenRemovals,
                          std::vector<Move> &candidates) {
  bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                                  MoveType::RSPR1Move) != typesBySpeed.end());
  bool delta_plus_present =
      (std::find(typesBySpeed.begin(), typesBySpeed.end(),
                 MoveType::DeltaPlusMove) != typesBySpeed.end());
  // add new possible moves to the candidate list
  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << "We have " << candidates.size()
              << " candidates before removing the old bad ones.\n";
  }
  updateOldCandidates(ann_network, chosenMove, candidates);
  if (takenRemovals.empty()) {
    std::vector<Node *> start_nodes = gatherStartNodes(ann_network, chosenMove);
    std::vector<Move> moreMoves =
        possibleMoves(ann_network, chosenMove.moveType, start_nodes,
                      rspr1_present, delta_plus_present, 0, best_max_distance);
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << "Adding " << moreMoves.size() << " candidates to the "
                << candidates.size() << " previous ones.\n ";
    }
    candidates.insert(std::end(candidates), std::begin(moreMoves),
                      std::end(moreMoves));
  }
  for (size_t i = 0; i < takenRemovals.size(); ++i) {
    updateOldCandidates(ann_network, takenRemovals[i], candidates);
  }
  removeBadCandidates(ann_network, candidates);
}

std::vector<Move> interleaveArcRemovals(
    AnnotatedNetwork &ann_network, PromisingStateQueue &psq,
    const std::vector<MoveType> &typesBySpeed, double *best_score,
    BestNetworkData *bestNetworkData, bool silent, bool print_progress) {
  std::vector<Move> takenRemovals;
  takenRemovals = fastIterationsMode(
      ann_network, psq, ann_network.options.max_rearrangement_distance,
      MoveType::ArcRemovalMove, typesBySpeed, best_score, bestNetworkData,
      silent, print_progress);
  if (!takenRemovals.empty()) {
    optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::NORMAL);
  }
  check_score_improvement(ann_network, best_score, bestNetworkData);
  return takenRemovals;
}

std::vector<Move> fastIterationsMode(AnnotatedNetwork &ann_network,
                                     PromisingStateQueue &psq,
                                     int best_max_distance, MoveType type,
                                     const std::vector<MoveType> &typesBySpeed,
                                     double *best_score,
                                     BestNetworkData *bestNetworkData,
                                     bool silent, bool print_progress) {
  assert(best_max_distance >= 0);
  std::vector<Move> acceptedMoves;

  NetworkState oldState = extract_network_state(ann_network);
  NetworkState bestState = extract_network_state(ann_network);

  std::vector<Move> candidates =
      getPossibleMoves(ann_network, typesBySpeed, type, 0, best_max_distance);
  prefilterCandidates(ann_network, psq, oldState, scoreNetwork(ann_network),
                      bestState, candidates, false, silent, print_progress);

  bool old_no_prefiltering = ann_network.options.no_prefiltering;
  ann_network.options.no_prefiltering = true;

  std::vector<Move> oldCandidates;

  bool tried_with_allnew = false;

  bool got_better = true;
  while (got_better) {
    got_better = false;
    oldCandidates = candidates;
    Move chosenMove = applyBestCandidate(
        ann_network, psq, candidates, best_score, bestNetworkData, false,
        ann_network.options.extreme_greedy, silent, print_progress);
    if (chosenMove.moveType != MoveType::INVALID) {
      extract_network_state(ann_network, oldState);
      // we accepted a move, thus score got better
      check_score_improvement(ann_network, best_score, bestNetworkData);
      acceptedMoves.emplace_back(chosenMove);
      tried_with_allnew = false;
      got_better = true;

      std::vector<Move> takenRemovals;
      if (isArcInsertion(chosenMove.moveType)) {
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << BLUE << "Trying to interleave arc removal moves.\n"
                    << RESET;
        }
        // try doing arc removal moves
        takenRemovals =
            interleaveArcRemovals(ann_network, psq, typesBySpeed, best_score,
                                  bestNetworkData, silent, print_progress);
        acceptedMoves.insert(acceptedMoves.end(), takenRemovals.begin(),
                             takenRemovals.end());
        if (!takenRemovals.empty()) {
          extract_network_state(ann_network, oldState);
        }
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << BLUE << "Back to arc insertions...\n" << RESET;
        }
      }

      // if we interleaved an arc insertion with some taken arc removal, it is
      // better to stop arc insertions for now and go on with some horizontal
      // moves first
      /*if (!takenRemovals.empty()) {
        ann_network.options.no_prefiltering = old_no_prefiltering;
        return acceptedMoves;
      }*/

      updateCandidateMoves(ann_network, typesBySpeed, best_max_distance,
                           chosenMove, takenRemovals, candidates);
      if (candidates.empty()) {
        // no old candidates to reuse. Thus,
        // completely gather new ones.
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << "no old candidates to reuse. Thus, completely gather "
                       "new ones.\n";
        }
        tried_with_allnew = true;
        candidates = getPossibleMoves(ann_network, typesBySpeed, type, 0,
                                      best_max_distance);
        oldCandidates.clear();
      }
      prefilterCandidates(ann_network, psq, oldState, scoreNetwork(ann_network),
                          bestState, candidates, false, silent, print_progress);
    } else {
      // score did not get better
      if (!tried_with_allnew && !acceptedMoves.empty()) {
        tried_with_allnew = true;
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          std::cout << "retrying with all new candidates\n";
        }
        candidates = getPossibleMoves(ann_network, typesBySpeed, type, 0,
                                      best_max_distance);
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&oldCandidates](const Move &move) {
                             return std::find(oldCandidates.begin(),
                                              oldCandidates.end(),
                                              move) != oldCandidates.end();
                           }),
            candidates.end());
        oldCandidates.clear();
        got_better = true;
      }
    }
  }

  ann_network.options.no_prefiltering = old_no_prefiltering;
  return acceptedMoves;
}  // namespace netrax

double slowIterationsMode(AnnotatedNetwork &ann_network,
                          PromisingStateQueue &psq, MoveType type,
                          int step_size,
                          const std::vector<MoveType> &typesBySpeed,
                          double *best_score, BestNetworkData *bestNetworkData,
                          bool silent, bool print_progress) {
  double old_score = scoreNetwork(ann_network);
  check_score_improvement(ann_network, best_score, bestNetworkData);
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
        getPossibleMoves(ann_network, typesBySpeed, type, min_dist, max_dist);
    applyBestCandidate(
        ann_network, psq, candidates, best_score, bestNetworkData, false,
        ann_network.options.extreme_greedy, silent, print_progress);
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

double fullSearch(AnnotatedNetwork &ann_network, PromisingStateQueue &psq,
                  MoveType type, const std::vector<MoveType> &typesBySpeed,
                  double *best_score, BestNetworkData *bestNetworkData,
                  bool silent, bool print_progress) {
  double old_score = scoreNetwork(ann_network);

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << BOLDBLUE;
    std::cout << "\nStarting full search for move type: " << toString(type)
              << "\n";
    std::cout << RESET;
  }

  int step_size = ann_network.options.step_size;

  // step 1: find best max distance
  int best_max_distance;
  if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove ||
      type == MoveType::DeltaMinusMove) {
    best_max_distance = ann_network.options.max_rearrangement_distance;
  } else {
    /*if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << toString(type) << " step1: find best max distance\n";
    }
    best_max_distance = findBestMaxDistance(ann_network, type, typesBySpeed,
                                            step_size, print_progress);*/
    best_max_distance = step_size;
  }

  // step 2: fast iterations mode, with the best max distance
  if (best_max_distance >= 0) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << "\n"
                << toString(type)
                << " step 2: fast iterations mode, with the best max distance "
                << best_max_distance << "\n";
    }
    fastIterationsMode(ann_network, psq, best_max_distance, type, typesBySpeed,
                       best_score, bestNetworkData, silent, print_progress);
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
    slowIterationsMode(ann_network, psq, type, step_size, typesBySpeed,
                       best_score, bestNetworkData, silent, print_progress);
  }

  old_score = scoreNetwork(ann_network);
  check_score_improvement(ann_network, best_score, bestNetworkData);

  return old_score;
}

}  // namespace netrax
