/*
 * TopologyOptimization.hpp
 *
 *  Created on: May 19, 2020
 *      Author: sarah
 */

#pragma once

#include <vector>
#include <limits>
#include "Moves.hpp"
#include "NetworkState.hpp"

namespace netrax {

struct AnnotatedNetwork;

void rankArcInsertionCandidates(AnnotatedNetwork& ann_network, std::vector<ArcInsertionMove>& candidates);

double forceApplyArcInsertion(AnnotatedNetwork& ann_network);

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, bool greedy = true, bool enforce_apply_move = false, bool silent = false, size_t max_iterations = std::numeric_limits<size_t>::max());
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types,NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse,  bool greedy=true, bool silent = false, size_t max_iterations = std::numeric_limits<size_t>::max());

void optimizeTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, bool greedy = true, bool silent = false, size_t max_iterations = std::numeric_limits<size_t>::max());
void optimizeTopology(AnnotatedNetwork &ann_network, MoveType& type, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, bool greedy=true, bool enforce_apply_move = false, bool silent = false, size_t max_iterations = std::numeric_limits<size_t>::max());

}
