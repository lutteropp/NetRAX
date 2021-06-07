#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../moves/MoveType.hpp"
#include "../optimization/NetworkState.hpp"
#include "ScoreImprovement.hpp"

namespace netrax {

double simanneal(AnnotatedNetwork &ann_network,
                 const std::vector<MoveType> &typesBySpeed, int min_radius,
                 int max_radius, double t_start,
                 BestNetworkData *bestNetworkData, bool silent = false);

}