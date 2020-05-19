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
#include "../graph/Moves.hpp"

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

double greedyHillClimbingStep(AnnotatedNetwork &ann_network, MoveType type, double old_score) {
    std::vector<std::unique_ptr<GeneralMove>> candidates = possibleMoves(ann_network, type);
    size_t best_idx = candidates.size();
    double best_score = old_score;
    for (size_t i = 0; i < candidates.size(); ++i) {
        performMove(ann_network, candidates[i].get());
        double new_logl = ann_network.raxml_treeinfo->loglh(true);
        double new_bic = bic(ann_network, new_logl);
        if (new_bic > best_score) {
            best_score = new_bic;
            best_idx = i;
        }
        undoMove(ann_network, candidates[i].get());
    }
    if (best_idx < candidates.size()) {
        performMove(ann_network, candidates[best_idx].get());
    }
    return best_score;
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double old_bic = bic(ann_network, old_logl);
    double new_bic = greedyHillClimbingStep(ann_network, type, old_bic);
    while (new_bic > old_bic) {
        old_bic = new_bic;
        new_bic = greedyHillClimbingStep(ann_network, type, old_bic);
    }
    return ann_network.old_logl;
}

double searchBetterTopology(AnnotatedNetwork &ann_network) {
    return greedyHillClimbingTopology(ann_network, MoveType::RNNIMove);
}

}
