/*
 * TopologyOptimization.hpp
 *
 *  Created on: May 19, 2020
 *      Author: sarah
 */

#pragma once

namespace netrax {

struct AnnotatedNetwork;
enum class MoveType;

double aic(AnnotatedNetwork &ann_network, double logl);
double bic(AnnotatedNetwork &ann_network, double logl);
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type);
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network);

}
