/*
 * TopologyOptimization.hpp
 *
 *  Created on: May 19, 2020
 *      Author: sarah
 */

#pragma once

namespace netrax {

struct AnnotatedNetwork;

double aic(AnnotatedNetwork &ann_network, double logl);
double bic(AnnotatedNetwork &ann_network, double logl);
double searchBetterTopology(AnnotatedNetwork &ann_network);

}
