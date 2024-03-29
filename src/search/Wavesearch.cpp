#include "Wavesearch.hpp"

#include "../DebugPrintFunctions.hpp"
#include "../NetraxOptions.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/PseudoLoglikelihood.hpp"
#include "../moves/Move.hpp"
#include "../optimization/NetworkState.hpp"
#include "../optimization/Optimization.hpp"
#include "CandidateSelection.hpp"
#include "Filtering.hpp"
#include "PromisingState.hpp"
#include "Scrambling.hpp"
#include "SimulatedAnnealing.hpp"

#include <algorithm>

#include "../colormod.h"  // namespace Color

namespace netrax {

void skipImpossibleTypes(AnnotatedNetwork &ann_network,
                         const std::vector<MoveType> &typesBySpeed,
                         unsigned int &type_idx) {
  while (ann_network.network.num_reticulations() == 0 &&
         isArcRemoval(typesBySpeed[type_idx])) {
    type_idx++;
    if (type_idx >= typesBySpeed.size()) {
      break;
    }
  }
  if (type_idx >= typesBySpeed.size()) {
    return;
  }
  while (ann_network.network.num_reticulations() ==
             ann_network.options.max_reticulations &&
         isArcInsertion(typesBySpeed[type_idx])) {
    type_idx++;
    if (type_idx >= typesBySpeed.size()) {
      break;
    }
  }
}

double optimizeEverythingRun(
    AnnotatedNetwork &ann_network, PromisingStateQueue &psq,
    const std::vector<MoveType> &typesBySpeed,
    const std::chrono::high_resolution_clock::time_point &start_time,
    BestNetworkData *bestNetworkData, bool skip_horizontal, bool silent,
    bool print_progress) {
  unsigned int type_idx = 0;
  unsigned int max_seconds = ann_network.options.timeout;
  double best_score = scoreNetwork(ann_network);

  unsigned int bad_rounds = 0;

  size_t old_moves_taken = ann_network.stats.totalMovesTaken();
  do {
    skipImpossibleTypes(ann_network, typesBySpeed, type_idx);

    while (skip_horizontal && type_idx < typesBySpeed.size() &&
           isHorizontalMove(typesBySpeed[type_idx])) {
      type_idx++;
      type_idx = type_idx % typesBySpeed.size();
    }
    skip_horizontal = false;

    double old_score = scoreNetwork(ann_network);
    double new_score =
        fullSearch(ann_network, psq, typesBySpeed[type_idx], typesBySpeed,
                   &best_score, bestNetworkData, silent, print_progress);

    if (new_score < old_score) {  // score got better
      new_score = scoreNetwork(ann_network);
      best_score = new_score;
      if (ann_network.stats.totalMovesTaken() > old_moves_taken) {
        old_moves_taken = ann_network.stats.totalMovesTaken();
      }
      bad_rounds = 0;
    } else {
      bad_rounds++;
    }
    assert(new_score <= old_score);
    type_idx++;
    type_idx = type_idx % typesBySpeed.size();

    if (max_seconds != 0) {
      auto act_time = std::chrono::high_resolution_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(act_time -
                                                           start_time)
              .count() >= max_seconds) {
        break;
      }
    }
  } while (bad_rounds < typesBySpeed.size());

  check_score_improvement(ann_network, &best_score, bestNetworkData);
  best_score = scoreNetwork(ann_network);

  return best_score;
}

void wavesearch_internal_loop(
    AnnotatedNetwork &ann_network, PromisingStateQueue &psq,
    BestNetworkData *bestNetworkData, const std::vector<MoveType> &typesBySpeed,
    double *best_score,
    const std::chrono::high_resolution_clock::time_point &start_time,
    bool silent, bool print_progress) {
  bool got_better = true;

  bool skip_horizontal = ann_network.options.good_start;

  check_score_improvement(ann_network, best_score, bestNetworkData);
  optimizeEverythingRun(ann_network, psq, typesBySpeed, start_time,
                        bestNetworkData, skip_horizontal, silent,
                        print_progress);
  check_score_improvement(ann_network, best_score, bestNetworkData);
}

void wavesearch_internal(
    AnnotatedNetwork &ann_network, PromisingStateQueue &psq,
    BestNetworkData *bestNetworkData, const std::vector<MoveType> &typesBySpeed,
    double *best_score,
    const std::chrono::high_resolution_clock::time_point &start_time,
    bool silent, bool print_progress) {
  double old_best_score = *best_score;

  wavesearch_internal_loop(ann_network, psq, bestNetworkData, typesBySpeed,
                           best_score, start_time, silent, print_progress);
  if (ann_network.options.retry > 0) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << MAGENTA << "The promising older candidates are: \n";
      for (size_t i = 0; i < psq.promising_states.size(); ++i) {
        std::cout << toString(psq.promising_states[i].move.moveType) << ": "
                  << psq.promising_states[i].target_bic << "\n";
      }
      std::cout << RESET;
    }
  }
  // Here we takine old other good configurations from the PSQ.
  for (size_t i = 0; i < ann_network.options.retry; ++i) {
    if (!hasPromisingStates(psq)) {
      break;
    }
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << MAGENTA
                << "\n\nRetrying search from a promising past state iteration "
                << i << "...\n"
                << RESET;
    }
    ann_network.options.retry = std::max(ann_network.options.retry - 1, 0);
    PromisingState pstate = getPromisingState(
        psq);  // TODO: This does an unneccessary copy operation. Fix this.
    applyPromisingState(ann_network, pstate, best_score, bestNetworkData, true,
                        silent);
    wavesearch_internal_loop(ann_network, psq, bestNetworkData, typesBySpeed,
                             best_score, start_time, silent, print_progress);
  }

  if (ann_network.options.enforce_extra_search) {
    bool got_better = true;
    // next, try enforcing some arc insertion
    while (got_better) {
      got_better = false;

      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << YELLOW << "Enforcing an arc insertion...\n" << RESET;
      }
      std::vector<Move> candidates;
      if (!ann_network.options.no_arc_insertion_moves) {
        candidates = possibleMoves(ann_network, MoveType::ArcInsertionMove,
                                   false, false);
      } else {
        candidates =
            possibleMoves(ann_network, MoveType::DeltaPlusMove, false, true);
      }
      NetworkState oldState = extract_network_state(ann_network);
      NetworkState bestState = extract_network_state(ann_network);
      double best_bic_prefilter = prefilterCandidates(
          ann_network, psq, oldState, scoreNetwork(ann_network), bestState,
          candidates, ann_network.options.extreme_greedy, silent,
          print_progress);
      applyBestCandidate(ann_network, psq, candidates, best_score,
                         bestNetworkData, true,
                         ann_network.options.extreme_greedy, best_bic_prefilter,
                         silent, print_progress);
      check_score_improvement(ann_network, best_score, bestNetworkData);
      if (*best_score < old_best_score) {
        got_better = true;
        old_best_score = *best_score;
      }
    }
  }
}

void wavesearch_main_internal(
    AnnotatedNetwork &ann_network, PromisingStateQueue &psq,
    BestNetworkData *bestNetworkData, const std::vector<MoveType> &typesBySpeed,
    NetworkState &start_state_to_reuse, NetworkState &best_state_to_reuse,
    double *best_score,
    const std::chrono::high_resolution_clock::time_point &start_time,
    bool silent, bool print_progress) {
  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << "Starting wavesearch with move types: ";
    for (size_t j = 0; j < typesBySpeed.size(); ++j) {
      std::cout << toString(typesBySpeed[j]);
      if (j + 1 < typesBySpeed.size()) {
        std::cout << ", ";
      }
    }
    std::cout << "\n";
  }

  if (!ann_network.options.scrambling_only) {
    wavesearch_internal(ann_network, psq, bestNetworkData, typesBySpeed,
                        best_score, start_time, silent, print_progress);
  } else {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << YELLOW
                << "Skipping initial inference and directly entering "
                   "scrambling mode.\n"
                << RESET;
    }
  }

  if (ann_network.options.scrambling > 0) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << CYAN;
      std::cout << "\nStarting scrambling phase...\n";
      std::cout << RESET;
    }
    unsigned int tries = 0;
    NetworkState bestState = extract_network_state(ann_network);
    double old_best_score = *best_score;
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      if (!silent)
        std::cout << CYAN << " Network before scrambling has BIC Score: "
                  << scoreNetwork(ann_network) << "\n"
                  << RESET;
    }
    while (tries < ann_network.options.scrambling) {
      apply_network_state(ann_network, bestState, true);
      double old_best_score_scrambling = scoreNetwork(ann_network);
      scrambleNetwork(ann_network, MoveType::RSPRMove,
                      ann_network.options.scrambling_radius);
      bool improved = true;
      while (improved) {
        improved = false;
        wavesearch_internal(ann_network, psq, bestNetworkData, typesBySpeed,
                            best_score, start_time, silent, print_progress);
        if (*best_score < old_best_score_scrambling) {
          old_best_score_scrambling = *best_score;
          improved = true;
        }
      }
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (!silent)
          std::cout << CYAN << " scrambling BIC: " << scoreNetwork(ann_network)
                    << "\n"
                    << RESET;
      }
      if (*best_score < old_best_score) {
        old_best_score = *best_score;
        extract_network_state(ann_network, bestState, true);
        tries = 0;
      } else {
        tries++;
      }
    }
    apply_network_state(ann_network, bestState, true);
  }
}

void wavesearch(AnnotatedNetwork &ann_network, BestNetworkData *bestNetworkData,
                const std::vector<MoveType> &typesBySpeed,
                const std::vector<MoveType> &typesBySpeedGoodStart, bool silent,
                bool print_progress) {
  PromisingStateQueue psq;
  NetworkState start_state_to_reuse = extract_network_state(ann_network);
  NetworkState best_state_to_reuse = extract_network_state(ann_network);
  auto start_time = std::chrono::high_resolution_clock::now();
  double best_score = std::numeric_limits<double>::infinity();
  // std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) <<
  // "\n\n";

  if (ann_network.options.sim_anneal) {
    simanneal(ann_network, typesBySpeed, 0, ann_network.options.step_size,
              ann_network.options.start_temperature, bestNetworkData, silent);
  } else {
    if (!ann_network.options.start_network_file
             .empty()) {  // don't waste time trying to first horizontally
                          // optimize the user-given start network
      optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::NORMAL);
      check_score_improvement(ann_network, &best_score, bestNetworkData);

      wavesearch_main_internal(ann_network, psq, bestNetworkData,
                               typesBySpeedGoodStart, start_state_to_reuse,
                               best_state_to_reuse, &best_score, start_time,
                               silent, print_progress);

    } else {
      optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::QUICK);
      check_score_improvement(ann_network, &best_score, bestNetworkData);
      wavesearch_main_internal(ann_network, psq, bestNetworkData, typesBySpeed,
                               start_state_to_reuse, best_state_to_reuse,
                               &best_score, start_time, silent, print_progress);
    }
  }
  optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::SLOW);
  check_score_improvement(ann_network, &best_score, bestNetworkData);
}

}  // namespace netrax