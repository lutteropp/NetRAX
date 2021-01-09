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
#include "../likelihood/LikelihoodComputation.hpp"
#include "../graph/Network.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../NetraxOptions.hpp"
#include "../io/NetworkIO.hpp"
#include "Moves.hpp"
#include "NetworkState.hpp"
#include "../utils.hpp"

namespace netrax {

double aic(double logl, double k) {
    return -2 * logl + 2 * k;
}
double aicc(double logl, double k, double n) {
    return aic(logl, k) + (2*k*k + 2*k) / (n - k - 1);
}
double bic(double logl, double k, double n) {
    return -2 * logl + k * log(n);
}

size_t get_param_count(AnnotatedNetwork& ann_network) {
    Network &network = ann_network.network;

    size_t param_count = ann_network.total_num_model_parameters;
    // reticulation probs as free parameters
    param_count += ann_network.network.num_reticulations();
    if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        assert(ann_network.fake_treeinfo->partition_count > 1);
        param_count += ann_network.fake_treeinfo->partition_count * network.num_branches();
    } else { // branch lengths are shared among partitions
        param_count += network.num_branches();
        if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
            assert(ann_network.fake_treeinfo->partition_count > 1);
            // each partition can scale the branch lengths by its own scaling factor
            param_count += ann_network.fake_treeinfo->partition_count - 1;
        }
    }
    return param_count;
}

size_t get_sample_size(AnnotatedNetwork& ann_network) {
    return ann_network.total_num_sites * ann_network.network.num_tips();
}

double aic(AnnotatedNetwork &ann_network, double logl) {
    return aic(logl, get_param_count(ann_network));
}

double aicc(AnnotatedNetwork &ann_network, double logl) {
    return aicc(logl, get_param_count(ann_network), get_sample_size(ann_network));
}

double bic(AnnotatedNetwork &ann_network, double logl) {
    return bic(logl, get_param_count(ann_network), get_sample_size(ann_network));
}

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


void assertBranchesWithinBounds(const AnnotatedNetwork& ann_network) {
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
}

void printExtensiveBICInfo(AnnotatedNetwork &ann_network) {
    std::cout << " brlen_linkage: ";
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        std::cout << "PLLMOD_COMMON_BRLEN_SCALED";
    } else if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        std::cout << "PLLMOD_COMMON_BRLEN_UNLINKED";
    } else {
        std::cout << "PLLMOD_COMMON_BRLEN_LINKED";
    }
    std::cout << "\n";
    std::cout << " network.num_reticulations: " << ann_network.network.num_reticulations() << "\n";
    std::cout << " network.num_branches: " << ann_network.network.num_branches() << "\n";
    std::cout << " ann_network.total_num_model_parameters: " << ann_network.total_num_model_parameters << "\n";
    std::cout << " ann_network.total_num_sites: " << ann_network.total_num_sites << "\n";
    std::cout << " ann_network.network.num_tips: " << ann_network.network.num_tips() << "\n";
    std::cout << " number of partitions in the MSA: " << ann_network.fake_treeinfo->partition_count << "\n";
    std::cout << " sample_size n: " << get_sample_size(ann_network) << "\n";
    std::cout << " param_count k: " << get_param_count(ann_network) << "\n";
    double logl = ann_network.raxml_treeinfo->loglh(true);
    std::cout << " logl: " << logl << "\n";
    std::cout << " bic: " << bic(ann_network, logl) << "\n";
}

void printOldDisplayedTrees(AnnotatedNetwork &ann_network) {
    std::cout << "Displayed trees info:\n";
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        for (size_t i = 0; i < ann_network.old_displayed_trees[p].size(); ++i) {
            std::cout << " partition " << p << ", displayed tree " << ann_network.old_displayed_trees[p][i].tree_idx << ":\n";
            std::cout << "   tree_logprob: " << ann_network.old_displayed_trees[p][i].tree_logprob << "\n";
            std::cout << "   tree_logl: " << ann_network.old_displayed_trees[p][i].tree_logl << "\n";
        }
    }
}

template<typename T>
double hillClimbingStep(AnnotatedNetwork &ann_network, std::vector<T> candidates, double old_score, bool greedy=true, bool randomizeCandidates=false, bool brlenopt_inside=true) {
    if (randomizeCandidates) {
        std::random_shuffle(candidates.begin(), candidates.end());
    }
    int max_iters = 1;
    int radius = 1;
    double start_logl = ann_network.raxml_treeinfo->loglh(true);
    NetworkState start_state = extract_network_state(ann_network);

    size_t best_idx = candidates.size();
    double best_score = old_score;

    bool verbose = false;

    NetworkState best_state = extract_network_state(ann_network);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (verbose) std::cout << toString(candidates[i].moveType) << " move " << i+1 << "/ " << candidates.size() << "\n";

        if (verbose) std::cout << "Extensive BIC info before applying current " << toString(candidates[i].moveType) << " move " << i+1 << "/ " << candidates.size() << ":\n";
        if (verbose) printExtensiveBICInfo(ann_network);
        if (verbose) printOldDisplayedTrees(ann_network);
        if (verbose) std::cout << "network before performing move: \n";
        if (verbose) std::cout << toExtendedNewick(ann_network) << "\n";

        //std::cout << " " << toString(candidates[i].moveType) << " " << i+1 << "/ " << candidates.size() << "\n";
        performMove(ann_network, candidates[i]);
        optimize_reticulations(ann_network, 100);
        
        if (brlenopt_inside) { // Do brlen optimization locally around the move
            if (verbose) std::cout << "AFTER MOVE, BUT BEFORE ANY OPT: \n";
            if (verbose) printOldDisplayedTrees(ann_network);
            if (verbose) std::cout << toExtendedNewick(ann_network) << "\n";

            std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, candidates[i]);
            optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);

            if (verbose) std::cout << "AFTER MOVE, AFTER BRLEN OPT: \n";
            if (verbose) printOldDisplayedTrees(ann_network);
            if (verbose) std::cout << toExtendedNewick(ann_network) << "\n";
            optimize_reticulations(ann_network, 100);
            if (verbose) std::cout << "AFTER MOVE, AFTER BRLEN OPT + SCALER OPT + RETICULATION OPT: \n";
            if (verbose) printOldDisplayedTrees(ann_network);
            if (verbose) std::cout << toExtendedNewick(ann_network) << "\n";

            if (candidates[i].moveType == MoveType::ArcRemovalMove || candidates[i].moveType == MoveType::ArcInsertionMove || candidates[i].moveType == MoveType::DeltaMinusMove || candidates[i].moveType == MoveType::DeltaPlusMove) {
                ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
                if (verbose) std::cout << "AFTER MOVE, AFTER BRLEN OPT + SCALER OPT + RETICULATION OPT + MODEL OPT: \n";
                if (verbose) printOldDisplayedTrees(ann_network);
                if (verbose) std::cout << toExtendedNewick(ann_network) << "\n";
            }
        }

        if (verbose) std::cout << "COMPUTING NEW LOGL\n";
        if (verbose) std::cout << toExtendedNewick(ann_network) << "\n";
        double new_logl = ann_network.raxml_treeinfo->loglh(true);

        if (candidates[i].moveType == MoveType::ArcInsertionMove || candidates[i].moveType == MoveType::DeltaPlusMove) {
            if (verbose) std::cout << "start_logl: " << start_logl << "\n";
            if (verbose) std::cout << "new_logl: " << new_logl << "\n";
            if (verbose) printOldDisplayedTrees(ann_network);
            assert(start_logl - new_logl < ann_network.options.lh_epsilon);
        }

        double new_score = bic(ann_network, new_logl);

        if (candidates[i].moveType == MoveType::ArcRemovalMove) {
            std::cout << "Tested " << toString(candidates[i]) << ", got " << new_score << "\n";
        }

        if (verbose) std::cout << "start_logl: " << start_logl << "\n";
        if (verbose) std::cout << "new_logl: " << new_logl << "\n";
        if (verbose) std::cout << "new_score: " << new_score << "\n";

        if (verbose) std::cout << "Extensive BIC info after applying current " << toString(candidates[i].moveType) << " move " << i+1 << "/ " << candidates.size() << ":\n";
        if (verbose) printExtensiveBICInfo(ann_network);

        bool foundBetterScore = false;
        if (new_score < best_score) {
            best_score = new_score;
            best_idx = i;
            if (brlenopt_inside) {
                best_state = extract_network_state(ann_network);
            }
            foundBetterScore = true;
        }
        undoMove(ann_network, candidates[i]);
        if (brlenopt_inside) {
            apply_network_state(ann_network, start_state);
            NetworkState act_state = extract_network_state(ann_network);
            assert(network_states_equal(start_state, act_state));
        }
        if (fabs(ann_network.raxml_treeinfo->loglh(true) - start_logl) >= ann_network.options.lh_epsilon) {
            std::cout << "new value: " << ann_network.raxml_treeinfo->loglh(true) << "\n";
            std::cout << "old value: " << start_logl << "\n";
        }
        assert(fabs(ann_network.raxml_treeinfo->loglh(true) - start_logl) < ann_network.options.lh_epsilon);
        if (greedy && foundBetterScore) {
            break;
        }
    }

    if (best_idx < candidates.size()) {
        performMove(ann_network, candidates[best_idx]);
        if (brlenopt_inside) {
            apply_network_state(ann_network, best_state);
        } else {
            std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, candidates[best_idx]);
            optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
            if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
                pllmod_algo_opt_brlen_scalers_treeinfo(ann_network.fake_treeinfo,
                                                                RAXML_BRLEN_SCALER_MIN,
                                                                RAXML_BRLEN_SCALER_MAX,
                                                                ann_network.options.brlen_min,
                                                                ann_network.options.brlen_max,
                                                                RAXML_PARAM_EPSILON);
            }
        }
        // just for debug, doing reticulation opt, full global brlen opt and model opt:
        optimize_reticulations(ann_network, 100);
        optimize_branches(ann_network, max_iters, radius);
        ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);

        best_score = bic(ann_network, ann_network.raxml_treeinfo->loglh(true));
        ann_network.stats.moves_taken[candidates[best_idx].moveType]++;

        std::cout << " Took " << toString(candidates[best_idx].moveType) << "\n";
        double logl = ann_network.raxml_treeinfo->loglh(true);
        double bic_score = bic(ann_network, logl);
        double aic_score = aic(ann_network, logl);
        double aicc_score = aicc(ann_network, logl);

        std::cout << "  Logl: " << logl << ", BIC: " << bic_score << ", AIC: " << aic_score << ", AICc: " << aicc_score <<  "\n";

        // just for debug
        /*double naive_logl = computeLoglikelihoodNaiveUtree(ann_network, 1, 1);
        double bic_naive = bic(ann_network, naive_logl);
        double aic_naive = aic(ann_network, naive_logl);
        double aicc_naive = aicc(ann_network, naive_logl);
        std::cout << "  Logl_naive: " << naive_logl << ", BIC_naive: " << bic_naive << ", AIC_naive: " << aic_naive << ", AICc_naive: " << aicc_naive << "\n";
        */

        std::cout << "  param_count: " << get_param_count(ann_network) << ", sample_size:" << get_sample_size(ann_network) << "\n";
        std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
        std::cout << toExtendedNewick(ann_network) << "\n";
    }
    return best_score;
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type, bool enforce_apply_move, size_t max_iterations) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double old_bic = bic(ann_network, old_logl);
    //std::cout << "start_logl: " << old_logl <<", start_bic: " << old_bic << "\n";
    std::cout << "Using move type: " << toString(type) << "\n";

    size_t act_iterations = 0;

    if (enforce_apply_move) {
        old_bic = std::numeric_limits<double>::max();
    }

    double new_score = old_bic;
    do {
        old_bic = new_score;

        switch (type) {
        case MoveType::RNNIMove:
            new_score = hillClimbingStep(ann_network, possibleRNNIMoves(ann_network), old_bic);
            break;
        case MoveType::RSPRMove:
            new_score = hillClimbingStep(ann_network, possibleRSPRMoves(ann_network), old_bic);
            break;
        case MoveType::RSPR1Move:
            new_score = hillClimbingStep(ann_network, possibleRSPR1Moves(ann_network), old_bic);
            break;
        case MoveType::HeadMove:
            new_score = hillClimbingStep(ann_network, possibleHeadMoves(ann_network), old_bic);
            break;
        case MoveType::TailMove:
            new_score = hillClimbingStep(ann_network, possibleTailMoves(ann_network), old_bic);
            break;
        case MoveType::ArcInsertionMove:
            new_score = hillClimbingStep(ann_network, possibleArcInsertionMoves(ann_network),
                    old_bic);
            break;
        case MoveType::DeltaPlusMove:
            new_score = hillClimbingStep(ann_network, possibleDeltaPlusMoves(ann_network),
                    old_bic);
            break;
        case MoveType::ArcRemovalMove:
            new_score = hillClimbingStep(ann_network, possibleArcRemovalMoves(ann_network),
                    old_bic);
            break;
        case MoveType::DeltaMinusMove:
            new_score = hillClimbingStep(ann_network, possibleDeltaMinusMoves(ann_network),
                    old_bic);
            break;
        default:
            throw std::runtime_error("Invalid move type");
        }

        /*
        if ((type == MoveType::ArcInsertionMove) || (type == MoveType::DeltaPlusMove)) {
            //doing reticulation opt, full global brlen opt and model opt:
            ann_network.raxml_treeinfo->optimize_branches(ann_network.options.lh_epsilon, 1);
            ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
            double new_logl;
            if (ann_network.options.use_nepal_prob_estimation) {
                new_logl = netrax::computeLoglikelihood(ann_network, 1, 1, true);
            } else {
                new_logl = netrax::optimize_reticulations(ann_network, 100);
            }
            new_score = bic(ann_network, new_logl);
        }
        */

       act_iterations++;
       if (act_iterations >= max_iterations) {
           break;
       }
    } while (old_bic - new_score > ann_network.options.lh_epsilon);
    return ann_network.raxml_treeinfo->loglh(true);
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types, size_t max_iterations) {
    unsigned int type_idx = 0;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
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
        new_logl = greedyHillClimbingTopology(ann_network, types[type_idx], max_iterations);
        new_score = bic(ann_network, new_logl);
        type_idx = (type_idx + 1) % types.size();
        moves_cnt++;

        if (old_bic - new_score > ann_network.options.lh_epsilon) {
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

double simulatedAnnealingTopology(AnnotatedNetwork &ann_network, MoveType type) {
    throw std::runtime_error("Not implemented yet");
}

double simulatedAnnealingTopology(AnnotatedNetwork &ann_network) {
    throw std::runtime_error("Not implemented yet");
}

}
