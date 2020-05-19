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

template<typename T>
std::vector<T> possibleMoves(AnnotatedNetwork &ann_network, MoveType &type) {
    switch (type) {
    case MoveType::RNNIMove:
        return possibleRNNIMoves(ann_network);
    case MoveType::RSPRMove:
        return possibleRSPRMoves(ann_network);
    case MoveType::HeadMove:
        return possibleHeadMoves(ann_network);
    case MoveType::TailMove:
        return possibleTailMoves(ann_network);
    case MoveType::RSPR1Move:
        return possibleRSPR1Moves(ann_network);
    case MoveType::ArcInsertionMove:
        return possibleArcInsertionMoves(ann_network);
    case MoveType::ArcRemovalMove:
        return possibleArcRemovalMoves(ann_network);
    case MoveType::DeltaPlusMove:
        return possibleDeltaPlusMoves(ann_network);
    case MoveType::DeltaMinusMove:
        return possibleDeltaMinusMoves(ann_network);
    }
}

template<typename T>
T randomMove(AnnotatedNetwork &ann_network, MoveType &type) {
    std::vector<T> candidates = possibleMoves<T>(ann_network, type);
    assert(!candidates.empty());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, candidates.size() - 1);
    return candidates[dist(ann_network.rng)];
}

double greedyHillClimbingStep(AnnotatedNetwork &ann_network, MoveType type) {
    return -1;
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double old_bic = bic(ann_network, old_logl);
    double new_bic = greedyHillClimbingStep(ann_network, type);
    while (new_bic > old_bic) {
        old_bic = new_bic;
        new_bic = greedyHillClimbingStep(ann_network, type);
    }
    return ann_network.old_logl;
}

double searchBetterTopology(AnnotatedNetwork &ann_network) {
    return greedyHillClimbingTopology(ann_network, MoveType::RNNIMove);
}

}
