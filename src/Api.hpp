/*
 * Api.hpp
 *
 *  Created on: Apr 24, 2020
 *      Author: sarah
 */

#pragma once

#include <string>

namespace netrax {

struct AnnotatedNetwork;
class NetraxOptions;

AnnotatedNetwork build_annotated_network(const NetraxOptions &options);
double computeLoglikelihood(AnnotatedNetwork &ann_network);
double updateReticulationProbs(AnnotatedNetwork &ann_network);
double optimizeModel(AnnotatedNetwork &ann_network);
double optimizeBranches(AnnotatedNetwork &ann_network);
double optimizeTopology(AnnotatedNetwork &ann_network);
void writeNetwork(AnnotatedNetwork &ann_network, const std::string& filepath);

}
