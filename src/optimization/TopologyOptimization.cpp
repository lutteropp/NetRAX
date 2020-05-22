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
double greedyHillClimbingStep(AnnotatedNetwork &ann_network, std::vector<T> candidates, double old_score) {
    size_t best_idx = candidates.size();
    double best_score = old_score;
    double best_logl;
    for (size_t i = 0; i < candidates.size(); ++i) {
        std::cout << "try move " << toString(candidates[i]) << "\n";
        performMove(ann_network, candidates[i]);
        double new_logl = ann_network.raxml_treeinfo->loglh(true);
        double new_bic = bic(ann_network, new_logl);
        if (new_bic < best_score) {
            best_score = new_bic;
            best_logl = new_logl;
            best_idx = i;
        }
        std::cout << "undo move " << toString(candidates[i]) << "\n";
        undoMove(ann_network, candidates[i]);
    }
    if (best_idx < candidates.size()) {
        performMove(ann_network, candidates[best_idx]);
        std::cout << "Accepting move " << toString(candidates[best_idx]) << " with old_score= " << old_score
                << ", best_score= " << best_score << ", best_logl= " << best_logl << "\n";
    }
    return best_score;
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double old_bic = bic(ann_network, old_logl);

    double new_bic = old_bic;
    do {
        old_bic = new_bic;

        switch (type) {
        case MoveType::RNNIMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleRNNIMoves(ann_network), old_bic);
            break;
        case MoveType::RSPRMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleRSPRMoves(ann_network), old_bic);
            break;
        case MoveType::RSPR1Move:
            new_bic = greedyHillClimbingStep(ann_network, possibleRSPR1Moves(ann_network), old_bic);
            break;
        case MoveType::HeadMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleHeadMoves(ann_network), old_bic);
            break;
        case MoveType::TailMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleTailMoves(ann_network), old_bic);
            break;
        case MoveType::ArcInsertionMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleArcInsertionMoves(ann_network), old_bic);
            break;
        case MoveType::DeltaPlusMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleDeltaPlusMoves(ann_network), old_bic);
            break;
        case MoveType::ArcRemovalMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleArcRemovalMoves(ann_network), old_bic);
            break;
        case MoveType::DeltaMinusMove:
            new_bic = greedyHillClimbingStep(ann_network, possibleDeltaMinusMoves(ann_network), old_bic);
            break;
        default:
            throw std::runtime_error("Invalid move type");
        }
    } while (new_bic < old_bic);
    return ann_network.old_logl;
}

}
