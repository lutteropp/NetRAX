#pragma once

#include <vector>

#include "../moves/Move.hpp"

namespace netrax {

struct BestNetworkData;

double prefilterCandidates(AnnotatedNetwork &ann_network,
                            const NetworkState &oldState,
                           std::vector<Move> &candidates, bool extreme_greedy,
                           bool silent, bool print_progress);

Move applyBestCandidate(AnnotatedNetwork &ann_network,
                        std::vector<Move> candidates, double *best_score,
                        BestNetworkData *bestNetworkData, bool enforce, bool extreme_greedy,
                        bool silent, bool print_progress);

double acceptMove(AnnotatedNetwork &ann_network, Move &move,
                  double *best_score, BestNetworkData *bestNetworkData,
                  bool silent = true);

}