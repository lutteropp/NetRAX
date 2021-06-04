#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../moves/Move.hpp"
#include "../moves/MoveType.hpp"
#include "ScoreImprovement.hpp"

namespace netrax {

struct BestNetworkData;

std::vector<Move> fastIterationsMode(AnnotatedNetwork &ann_network,
                                     int min_radius, int max_radius,
                                     MoveType type,
                                     const std::vector<MoveType> &typesBySpeed,
                                     double *best_score,
                                     BestNetworkData *bestNetworkData,
                                     bool silent, bool print_progress);

double fullSearch(AnnotatedNetwork &ann_network, MoveType type,
                  const std::vector<MoveType> &typesBySpeed, double *best_score,
                  BestNetworkData *bestNetworkData, bool silent,
                  bool print_progress);

void updateOldCandidates(AnnotatedNetwork &ann_network, const Move &chosenMove,
                         std::vector<Move> &candidates);

}  // namespace netrax