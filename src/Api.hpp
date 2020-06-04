/*
 * Api.hpp
 *
 *  Created on: Apr 24, 2020
 *      Author: sarah
 */

#pragma once

#include <string>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

struct AnnotatedNetwork;
class NetraxOptions;

AnnotatedNetwork build_annotated_network(const NetraxOptions &options);
AnnotatedNetwork build_annotated_network_from_string(const NetraxOptions &options,
        const std::string &newickString);
AnnotatedNetwork build_annotated_network_from_utree(const NetraxOptions &options,
        const pll_utree_t &utree);
AnnotatedNetwork build_random_annotated_network(const NetraxOptions &options,
        unsigned int start_reticulations = 0);
AnnotatedNetwork build_parsimony_annotated_network(const NetraxOptions &options,
        unsigned int start_reticulations = 0);
AnnotatedNetwork build_best_raxml_annotated_network(const NetraxOptions &options,
        unsigned int start_reticulations = 0);
double computeLoglikelihood(AnnotatedNetwork &ann_network);
double updateReticulationProbs(AnnotatedNetwork &ann_network);
double optimizeModel(AnnotatedNetwork &ann_network);
double optimizeBranches(AnnotatedNetwork &ann_network);
double optimizeTopology(AnnotatedNetwork &ann_network);
double optimizeEverything(AnnotatedNetwork &ann_network);
void writeNetwork(AnnotatedNetwork &ann_network, const std::string &filepath);

}
