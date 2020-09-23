/*
 * TopologyOptimization.hpp
 *
 *  Created on: May 19, 2020
 *      Author: sarah
 */

#pragma once

#include <vector>
#include "Moves.hpp"

namespace netrax {

struct AnnotatedNetwork;

double aic(AnnotatedNetwork &ann_network, double logl);
double bic(AnnotatedNetwork &ann_network, double logl);
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type);
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types);

double simulatedAnnealingTopology(AnnotatedNetwork &ann_network, MoveType type);
double simulatedAnnealingTopology(AnnotatedNetwork &ann_network);
}
