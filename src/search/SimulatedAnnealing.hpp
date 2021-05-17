#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "ScoreImprovement.hpp"
#include "../moves/MoveType.hpp"
#include "../optimization/NetworkState.hpp"

namespace netrax {

double simanneal(AnnotatedNetwork& ann_network, double t_start, bool rspr1_present, bool delta_plus_present, MoveType type, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, BestNetworkData* bestNetworkData, bool silent = false);

}