#pragma once

#include <vector>

#include "../moves/Move.hpp"

namespace netrax {

struct BestNetworkData;

double prefilterCandidates(AnnotatedNetwork &ann_network,
                           std::vector<Move> &candidates, bool silent = true,
                           bool print_progress = true,
                           bool need_best_bic = false);

Move applyBestCandidate(AnnotatedNetwork &ann_network,
                        std::vector<Move> candidates, double *best_score,
                        BestNetworkData *bestNetworkData, bool enforce,
                        bool silent);

double acceptMove(AnnotatedNetwork &ann_network, Move &move,
                  double expected_bic, const NetworkState &state,
                  double *best_score, BestNetworkData *bestNetworkData,
                  bool silent = true);

}