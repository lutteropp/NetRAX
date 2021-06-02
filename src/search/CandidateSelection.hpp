#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../moves/Move.hpp"
#include "../moves/MoveType.hpp"
#include "ScoreImprovement.hpp"

namespace netrax {

Move applyBestCandidate(AnnotatedNetwork &ann_network,
                        std::vector<Move> candidates, double *best_score,
                        BestNetworkData *bestNetworkData, bool enforce,
                        bool silent);

double fullSearch(AnnotatedNetwork &ann_network, MoveType type,
                  const std::vector<MoveType> &typesBySpeed, double *best_score,
                  BestNetworkData *bestNetworkData, bool silent);

void updateOldCandidates(AnnotatedNetwork &ann_network, const Move &chosenMove,
                         std::vector<Move> &candidates);

}  // namespace netrax