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

namespace netrax {

struct AnnotatedNetwork;

double aic(AnnotatedNetwork &ann_network, double logl);
double bic(AnnotatedNetwork &ann_network, double logl);
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type, bool greedy = true, bool enforce_apply_move = false, size_t max_iterations = std::numeric_limits<size_t>::max());
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types, bool greedy=true, size_t max_iterations = std::numeric_limits<size_t>::max());

double simulatedAnnealingTopology(AnnotatedNetwork &ann_network, MoveType type);
double simulatedAnnealingTopology(AnnotatedNetwork &ann_network);
}
