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
std::vector<T> possibleMoves(AnnotatedNetwork &ann_network, MoveType moveType) {
    switch (moveType) {
    case MoveType::RNNIMove:
        return possibleRNNIMoves(ann_network);
    case MoveType::RSPRMove:
        return possibleRSPRMoves(ann_network);
    case MoveType::RSPR1Move:
        return possibleRSPR1Moves(ann_network);
    case MoveType::HeadMove:
        return possibleHeadMoves(ann_network);
    case MoveType::TailMove:
        return possibleTailMoves(ann_network);
    case MoveType::ArcInsertionMove:
        return possibleArcInsertionMoves(ann_network);
    case MoveType::DeltaPlusMove:
        return possibleDeltaPlusMoves(ann_network);
    case MoveType::ArcRemovalMove:
        return possibleArcRemovalMoves(ann_network);
    case MoveType::DeltaMinusMove:
        return possibleDeltaMinusMoves(ann_network);
    }
    throw std::runtime_error("Invalid move type");
}

template<typename T>
double greedyHillClimbingStep(AnnotatedNetwork &ann_network, std::vector<T> &candidates, double old_score) {
    size_t best_idx = candidates.size();
    double best_score = old_score;
    for (size_t i = 0; i < candidates.size(); ++i) {
        performMove(ann_network, candidates[i]);
        double new_logl = ann_network.raxml_treeinfo->loglh(true);
        double new_bic = bic(ann_network, new_logl);
        if (new_bic > best_score) {
            best_score = new_bic;
            best_idx = i;
        }
        undoMove(ann_network, candidates[i]);
    }
    if (best_idx < candidates.size()) {
        performMove(ann_network, candidates[best_idx]);
    }
    return best_score;
}

template<typename T>
double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType &type) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double old_bic = bic(ann_network, old_logl);

    std::vector<T> candidates = possibleMoves<T>(ann_network, type);
    double new_bic = greedyHillClimbingStep<T>(ann_network, candidates, old_bic);
    while (new_bic > old_bic) {
        old_bic = new_bic;
        candidates = possibleMoves<T>(ann_network, type);
        new_bic = greedyHillClimbingStep<T>(ann_network, candidates, old_bic);
    }
    return ann_network.old_logl;
}

double searchBetterTopologyGreedy(AnnotatedNetwork &ann_network, MoveType type) {
    switch (type) {
    case MoveType::RNNIMove:
        return greedyHillClimbingTopology<RNNIMove>(ann_network, type);
    case MoveType::RSPRMove:
        return greedyHillClimbingTopology<RSPRMove>(ann_network, type);
    case MoveType::RSPR1Move:
        return greedyHillClimbingTopology<RSPRMove>(ann_network, type);
    case MoveType::HeadMove:
        return greedyHillClimbingTopology<RSPRMove>(ann_network, type);
    case MoveType::TailMove:
        return greedyHillClimbingTopology<RSPRMove>(ann_network, type);
    case MoveType::ArcInsertionMove:
        return greedyHillClimbingTopology<ArcInsertionMove>(ann_network, type);
    case MoveType::DeltaPlusMove:
        return greedyHillClimbingTopology<ArcInsertionMove>(ann_network, type);
    case MoveType::ArcRemovalMove:
        return greedyHillClimbingTopology<ArcRemovalMove>(ann_network, type);
    case MoveType::DeltaMinusMove:
        return greedyHillClimbingTopology<ArcRemovalMove>(ann_network, type);
    }
    throw std::runtime_error("Ivalid move type");
}

}
