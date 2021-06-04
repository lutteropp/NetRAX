#pragma once

#include <vector>

#include "../moves/Move.hpp"

namespace netrax {

struct AnnotatedNetwork;
struct BestNetworkData;
struct NetworkState;

double prefilterCandidates(AnnotatedNetwork &ann_network,
                           const NetworkState &oldState,
                           NetworkState &bestState,
                           std::vector<Move> &candidates, bool extreme_greedy,
                           bool silent, bool print_progress);

Move applyBestCandidate(AnnotatedNetwork &ann_network,
                        std::vector<Move> candidates, double *best_score,
                        BestNetworkData *bestNetworkData, bool enforce,
                        bool extreme_greedy, bool silent, bool print_progress);

double acceptMove(AnnotatedNetwork &ann_network, Move &move,
                  const NetworkState &bestState, double *best_score,
                  BestNetworkData *bestNetworkData, bool silent = true);

}  // namespace netrax