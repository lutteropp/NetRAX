#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "ScoreImprovement.hpp"
#include "../moves/MoveType.hpp"

namespace netrax {

double applyBestCandidate(AnnotatedNetwork& ann_network, MoveType type, const std::vector<MoveType>& typesBySpeed, double* best_score, BestNetworkData* bestNetworkData, bool enforce, bool silent, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());

double fullSearch(AnnotatedNetwork& ann_network, MoveType type, const std::vector<MoveType>& typesBySpeed, double* best_score, BestNetworkData* bestNetworkData, bool silent);


}