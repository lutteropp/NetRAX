/*
 * TopologyOptimization.cpp
 *
 *  Created on: May 19, 2020
 *      Author: sarah
 */

#include "TopologyOptimization.hpp"
#include <cmath>

#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../graph/Network.hpp"
#include "../graph/Moves.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "BranchLengthOptimization.hpp"
#include "../io/NetworkIO.hpp"

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

std::vector<std::vector<double> > extract_brlens(AnnotatedNetwork &ann_network) {
    std::vector<std::vector<double> > res;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t n_partitions = 1;
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    res.resize(n_partitions);
    for (size_t p = 0; p < n_partitions; ++p) {
        res[p].resize(ann_network.network.edges.size());
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            res[p][pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    }
    return res;
}

void apply_brlens(AnnotatedNetwork &ann_network, const std::vector<std::vector<double> > &old_brlens) {
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t n_partitions = 1;
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    for (size_t p = 0; p < n_partitions; ++p) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            if (ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] != old_brlens[p][pmatrix_index]) {
                ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] = old_brlens[p][pmatrix_index];
                ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
                invalidateHigherCLVs(ann_network, getTarget(ann_network.network, &ann_network.network.edges[i]), true);
            }
        }
    }
}

template<typename T>
double greedyHillClimbingStep(AnnotatedNetwork &ann_network, std::vector<T> candidates, double old_score) {
    //std::cout << "greedyHillclimbingStep called\n";
    size_t best_idx = candidates.size();
    double best_score = old_score;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    size_t old_reticulation_count = ann_network.network.num_reticulations();
    double best_logl;
    //std::vector<std::vector<double> > old_brlens = extract_brlens(ann_network);
    //std::vector<std::vector<double> > best_brlens;
    //int radius = 1;
    //int max_iters = ann_network.options.brlen_smoothings;
    for (size_t i = 0; i < candidates.size(); ++i) {
        //std::cout << exportDebugInfo(ann_network.network);
        std::cout << "try move " << toString(candidates[i]) << "\n";
        performMove(ann_network, candidates[i]);
        //std::cout << exportDebugInfo(ann_network.network);
        //std::cout << "logl after perform move: " << ann_network.raxml_treeinfo->loglh(true) <<"\n";
        //std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, candidates[i]);
        //optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
        double new_logl = ann_network.raxml_treeinfo->loglh(true);
        double new_bic = bic(ann_network, new_logl);
        //std::cout << "bic after perform move: " << new_bic <<"\n";
        if (new_bic < best_score) {
            best_score = new_bic;
            best_logl = new_logl;
            best_idx = i;
            //best_brlens = extract_brlens(ann_network);
            size_t new_reticulation_count = ann_network.network.num_reticulations();
            std::cout << "prev_logl: " << old_logl << ", prev_bic: " << best_score << ", new_logl: " << new_logl << ", new_score: " << new_bic << "\n";
            assert(old_reticulation_count > new_reticulation_count || new_logl > old_logl);
            old_logl = best_logl;
            old_reticulation_count = ann_network.network.num_reticulations();
        }
        std::cout << "undo move " << toString(candidates[i]) << "\n";
        undoMove(ann_network, candidates[i]);
        //std::cout << "logl after undo move: " << ann_network.raxml_treeinfo->loglh(true) <<"\n";
        //apply_brlens(ann_network, old_brlens);
    }
    if (best_idx < candidates.size()) {
        performMove(ann_network, candidates[best_idx]);
        //apply_brlens(ann_network, best_brlens);
        // optimize reticulation probs and model after a move has been accepted
        //netrax::computeLoglikelihood(ann_network, 1, 1, true);
        best_logl = ann_network.raxml_treeinfo->loglh(true);
        //best_logl = ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
        //best_logl = optimize_branches(ann_network, max_iters, radius);

        //std::cout << "Accepting move " << toString(candidates[best_idx]) << " with old_score= " << old_score
          //      << ", best_score= " << best_score << ", best_logl= " << best_logl << "\n";
        assertReticulationProbs(ann_network);
    }
    return best_score;
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double old_bic = bic(ann_network, old_logl);
    //std::cout << "start_logl: " << old_logl <<", start_bic: " << old_bic << "\n";

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
    return ann_network.raxml_treeinfo->loglh(true);
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network) {
    //std::vector<MoveType> types = { MoveType::DeltaMinusMove, MoveType::RNNIMove, MoveType::RSPR1Move,
    //        MoveType::DeltaPlusMove };

    std::vector<MoveType> types = { MoveType::ArcRemovalMove, MoveType::RNNIMove, MoveType::RSPRMove,
            MoveType::ArcInsertionMove };
    unsigned int type_idx = 0;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double new_logl = old_logl;
    double old_bic = bic(ann_network, old_logl);
    double new_bic = old_bic;
    unsigned int moves_cnt = 0;
    do {
        //std::cout << toExtendedNewick(ann_network.network) << "\n";
        if (new_bic < old_bic) {
            std::cout << "Improved bic from " << old_bic << " to " << new_bic << "\n";
            old_bic = new_bic;
            moves_cnt = 0;
        }

        std::cout << "Using move type: " << toString(types[type_idx]) << "\n";
        new_logl = greedyHillClimbingTopology(ann_network, types[type_idx]);
        new_bic = bic(ann_network, new_logl);
        type_idx = (type_idx + 1) % types.size();
        moves_cnt++;
        if (ann_network.network.num_reticulations() == 0
                && (types[type_idx] == MoveType::DeltaMinusMove || types[type_idx] == MoveType::ArcRemovalMove)) {
            type_idx = (type_idx + 1) % types.size();
            moves_cnt++;
        }
    } while ((new_bic < old_bic) || (moves_cnt < types.size()));
    return new_logl;
}

}
