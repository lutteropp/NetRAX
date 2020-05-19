/*
 * TopologyOptimization.cpp
 *
 *  Created on: May 19, 2020
 *      Author: sarah
 */

#include "TopologyOptimization.hpp"
#include <cmath>

#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/Network.hpp"

namespace netrax {

double aic(double logl, size_t k) {
    return -2 * logl + 2 * k;
}
double bic(double logl, size_t k, size_t n) {
    return -2 * logl + k * log(n);
}

double aic(AnnotatedNetwork &ann_network, double logl) {
    Network &network = ann_network.network;
    bool unlinked_mode = (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t multiplier = (unlinked_mode) ? 1 : ann_network.fake_treeinfo->partition_count;
    size_t param_count = multiplier * network.num_branches() + ann_network.total_num_model_parameters;
    return aic(logl, param_count);
}

double bic(AnnotatedNetwork &ann_network, double logl) {
    Network &network = ann_network.network;
    bool unlinked_mode = (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t multiplier = (unlinked_mode) ? 1 : ann_network.fake_treeinfo->partition_count;
    size_t param_count = multiplier * network.num_branches() + ann_network.total_num_model_parameters;
    size_t num_sites = ann_network.total_num_sites;
    return bic(logl, param_count, num_sites);
}

double searchBetterTopology(AnnotatedNetwork &ann_network) {
    return -1;
}

}
