/*
 * TopologyOptimization.cpp
 *
 *  Created on: May 19, 2020
 *      Author: sarah
 */

#include "TopologyOptimization.hpp"

#include <stddef.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>
#include <raxml-ng/TreeInfo.hpp>
#include "../DebugPrintFunctions.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../optimization/Moves.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/ReticulationOptimization.hpp"
#include "../optimization/ModelOptimization.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../graph/Network.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../NetraxOptions.hpp"
#include "../io/NetworkIO.hpp"
#include "Moves.hpp"
#include "NetworkState.hpp"
#include "../utils.hpp"

namespace netrax {

template<typename T>
bool wantedMove(T *move) {
    if (move->moveType == MoveType::ArcRemovalMove) {
        ArcRemovalMove *m = (ArcRemovalMove*) move;
        if (m->a_clv_index == 11 && m->b_clv_index == 21 && m->c_clv_index == 18
                && m->d_clv_index == 6 && m->u_clv_index == 12 && m->v_clv_index == 19) {
            return true;
        }
    }
    return false;
}


bool assertBranchesWithinBounds(const AnnotatedNetwork& ann_network) {
    double min_brlen = ann_network.options.brlen_min;
    double max_brlen = ann_network.options.brlen_max;
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
        double w = ann_network.network.edges_by_index[pmatrix_index]->length;
        double w2 = ann_network.fake_treeinfo->branch_lengths[0][pmatrix_index];
        assert(w==w2);
        assert(w>=min_brlen);
        assert(w<=max_brlen);
    }
    return true;
}

bool isComplexityChanging(MoveType& moveType) {
    return (moveType == MoveType::ArcRemovalMove || moveType == MoveType::ArcInsertionMove || moveType == MoveType::DeltaMinusMove || moveType == MoveType::DeltaPlusMove);
}

template<typename T>
double hillClimbingStep(AnnotatedNetwork &ann_network, std::vector<T> candidates, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, double old_score, TopoSettings topoSettings) {
    if (candidates.empty()) {
        if (!topoSettings.silent) std::cout << "empty list of candidates\n";
        return bic(ann_network, computeLoglikelihood(ann_network));
    } else {
        if (!topoSettings.silent) std::cout << candidates.size() << " candidates\n";
    }

    bool complexityChanging = isComplexityChanging(candidates[0].moveType);
    
    if (topoSettings.randomize_candidates) {
        std::random_shuffle(candidates.begin(), candidates.end());
    }

    extract_network_state(ann_network, start_state_to_reuse, complexityChanging);

    double brlen_smooth_factor = 0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = 1;
    double start_logl = computeLoglikelihood(ann_network);

    size_t best_idx = candidates.size();
    double best_score = old_score;

    extract_network_state(ann_network, best_state_to_reuse, complexityChanging);

    for (size_t i = 0; i < candidates.size(); ++i) {
        T move = candidates[i];
        performMove(ann_network, move);
        if (move.moveType == MoveType::ArcRemovalMove || move.moveType == MoveType::DeltaMinusMove) { // because arc removals change the reticulation indices, breaking the stored reticulationChoices in the displayed trees
            computeLoglikelihood(ann_network, 0, 1);
        }
        // Do brlen optimization locally around the move
        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        //add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
        optimizeReticulationProbs(ann_network);
        optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
        //optimizeReticulationProbs(ann_network);

        double new_logl = computeLoglikelihood(ann_network);
        double new_score = bic(ann_network, new_logl);

        bool foundBetterScore = false;
        if (new_score < old_score) {
            best_score = new_score;
            best_idx = i;
            extract_network_state(ann_network, best_state_to_reuse, complexityChanging);
            foundBetterScore = true;
        }
        if (!isComplexityChanging(move.moveType)) {
            undoMove(ann_network, move);
        }
        apply_network_state(ann_network, start_state_to_reuse, complexityChanging);
        assert(network_states_equal(start_state_to_reuse, extract_network_state(ann_network, complexityChanging)));

        if (fabs(computeLoglikelihood(ann_network) - start_logl) >= ann_network.options.lh_epsilon) {
            double new_value = computeLoglikelihood(ann_network);
            if (complexityChanging) {
                std::cout << "we are complexity changing\n";
            }
            assert(network_states_equal(start_state_to_reuse, extract_network_state(ann_network, complexityChanging)));
            throw std::runtime_error("Rolling back did not work correctly");
        }
        assert(fabs(computeLoglikelihood(ann_network) - start_logl) < ann_network.options.lh_epsilon);
        if (topoSettings.greedy && foundBetterScore) {
            break;
        }
    }

    if (best_idx < candidates.size()) {
        T bestMove = candidates[best_idx];
        performMove(ann_network, bestMove);

        apply_network_state(ann_network, best_state_to_reuse, complexityChanging);

        best_score = bic(ann_network, computeLoglikelihood(ann_network));
        ann_network.stats.moves_taken[candidates[best_idx].moveType]++;

        if (!topoSettings.silent) std::cout << " Took " << toString(candidates[best_idx].moveType) << "\n";
        double logl = computeLoglikelihood(ann_network);
        double bic_score = bic(ann_network, logl);
        double aic_score = aic(ann_network, logl);
        double aicc_score = aicc(ann_network, logl);

        if (!topoSettings.silent) std::cout << "  Logl: " << logl << ", BIC: " << bic_score << ", AIC: " << aic_score << ", AICc: " << aicc_score <<  "\n";
        if (!topoSettings.silent) std::cout << "  param_count: " << get_param_count(ann_network) << ", sample_size:" << get_sample_size(ann_network) << "\n";
        if (!topoSettings.silent) std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
        if (!topoSettings.silent) std::cout << toExtendedNewick(ann_network) << "\n";
    }
    return best_score;
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, TopoSettings topoSettings, size_t max_iterations) {
    double old_logl = computeLoglikelihood(ann_network);
    double old_bic = bic(ann_network, old_logl);
    //std::cout << "start_logl: " << old_logl <<", start_bic: " << old_bic << "\n";
    if (!topoSettings.silent) std::cout << "Using move type: " << toString(type) << "\n";

    size_t act_iterations = 0;

    if (topoSettings.enforce_apply_move) {
        if (!topoSettings.silent) std::cout << "enforce apply move is active\n";
        old_bic = std::numeric_limits<double>::max();
    }

    double new_score = old_bic;
    do {
        old_bic = new_score;

        switch (type) {
        case MoveType::RNNIMove:
            new_score = hillClimbingStep(ann_network, possibleRNNIMoves(ann_network), start_state_to_reuse, best_state_to_reuse, old_bic, topoSettings);
            break;
        case MoveType::RSPRMove:
            new_score = hillClimbingStep(ann_network, possibleRSPRMoves(ann_network), start_state_to_reuse, best_state_to_reuse, old_bic, topoSettings);
            break;
        case MoveType::RSPR1Move:
            new_score = hillClimbingStep(ann_network, possibleRSPR1Moves(ann_network), start_state_to_reuse, best_state_to_reuse, old_bic, topoSettings);
            break;
        case MoveType::HeadMove:
            new_score = hillClimbingStep(ann_network, possibleHeadMoves(ann_network), start_state_to_reuse, best_state_to_reuse, old_bic, topoSettings);
            break;
        case MoveType::TailMove:
            new_score = hillClimbingStep(ann_network, possibleTailMoves(ann_network), start_state_to_reuse, best_state_to_reuse, old_bic, topoSettings);
            break;
        case MoveType::ArcInsertionMove:
            new_score = hillClimbingStep(ann_network, possibleArcInsertionMoves(ann_network), start_state_to_reuse, best_state_to_reuse, 
                    old_bic, topoSettings);
            break;
        case MoveType::DeltaPlusMove:
            new_score = hillClimbingStep(ann_network, possibleDeltaPlusMoves(ann_network), start_state_to_reuse, best_state_to_reuse,
                    old_bic, topoSettings);
            break;
        case MoveType::ArcRemovalMove:
            new_score = hillClimbingStep(ann_network, possibleArcRemovalMoves(ann_network), start_state_to_reuse, best_state_to_reuse,
                    old_bic, topoSettings);
            break;
        case MoveType::DeltaMinusMove:
            new_score = hillClimbingStep(ann_network, possibleDeltaMinusMoves(ann_network), start_state_to_reuse, best_state_to_reuse,
                    old_bic, topoSettings);
            break;
        default:
            throw std::runtime_error("Invalid move type");
        }

       act_iterations++;
       if (act_iterations >= max_iterations) {
           break;
       }
    } while (old_bic - new_score > ann_network.options.score_epsilon);
    return computeLoglikelihood(ann_network);
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, TopoSettings topoSettings, size_t max_iterations) {
    unsigned int type_idx = 0;
    double old_logl = computeLoglikelihood(ann_network);
    double new_logl = old_logl;
    double old_bic = bic(ann_network, old_logl);
    double new_score = old_bic;
    unsigned int moves_cnt = 0;
    size_t act_iterations = 0;
    do {
        if (ann_network.network.num_reticulations() == 0
              && (types[type_idx] == MoveType::DeltaMinusMove || types[type_idx] == MoveType::ArcRemovalMove)) {
            type_idx = (type_idx + 1) % types.size();
            moves_cnt++;
        }
        //std::cout << "Using move type: " << toString(types[type_idx]) << "\n";
        new_logl = greedyHillClimbingTopology(ann_network, types[type_idx], start_state_to_reuse, best_state_to_reuse, topoSettings, 1);
        new_score = bic(ann_network, new_logl);
        type_idx = (type_idx + 1) % types.size();
        moves_cnt++;

        if (old_bic - new_score > ann_network.options.score_epsilon) {
            //std::cout << "Improved bic from " << old_bic << " to " << new_score << "\n";
            old_bic = new_score;
            moves_cnt = 0;
        }
        act_iterations++;
        if (act_iterations >= max_iterations) {
            break;
        }
    } while (moves_cnt < types.size());
    return new_logl;
}

bool logl_same_after_recompute(AnnotatedNetwork& ann_network) {
    /*double incremental = netrax::computeLoglikelihood(ann_network, 1, 1);
    double normal = netrax::computeLoglikelihood(ann_network, 0, 1);
    if (incremental != normal) {
        std::cout << "incremental: " << incremental << "\n";
        std::cout << "normal: " << normal << "\n";
    }
    return (incremental == normal);*/
    return true;
}

/**
 * Re-infers the topology of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, TopoSettings topoSettings, size_t max_iterations) {
    assert(logl_same_after_recompute(ann_network));
    double old_score = scoreNetwork(ann_network);
    greedyHillClimbingTopology(ann_network, types, start_state_to_reuse, best_state_to_reuse, topoSettings, max_iterations);
    double new_score = scoreNetwork(ann_network);
    if (!topoSettings.silent) std::cout << "BIC after topology optimization: " << new_score << "\n";
    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

/**
 * Re-infers the topology of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeTopology(AnnotatedNetwork &ann_network, MoveType& type, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, TopoSettings topoSettings, size_t max_iterations) {
    assert(logl_same_after_recompute(ann_network));
    double old_score = scoreNetwork(ann_network);
    greedyHillClimbingTopology(ann_network, type, start_state_to_reuse, best_state_to_reuse, topoSettings, max_iterations);
    double new_score = scoreNetwork(ann_network);
    if (!topoSettings.silent) std::cout << "BIC after topology optimization: " << new_score << "\n";

    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

}
