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
#include "Moves.hpp"

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
    size_t param_count = multiplier * network.num_branches()
            + ann_network.total_num_model_parameters;
    return aic(logl, param_count);
}

double bic(AnnotatedNetwork &ann_network, double logl) {
    Network &network = ann_network.network;
    bool unlinked_mode = (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t multiplier = (unlinked_mode) ? 1 : ann_network.fake_treeinfo->partition_count;
    size_t param_count = multiplier * network.num_branches()
            + ann_network.total_num_model_parameters;
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

std::vector<std::vector<double> > extract_brprobs(AnnotatedNetwork &ann_network) {
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
            res[p][pmatrix_index] = ann_network.network.edges[i].prob;
        }
    }
    return res;
}

void apply_brlens(AnnotatedNetwork &ann_network,
        const std::vector<std::vector<double> > &old_brlens) {
    std::vector<bool> visited(ann_network.network.nodes.size(), false);
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t n_partitions = 1;
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    for (size_t p = 0; p < n_partitions; ++p) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            if (ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index]
                    != old_brlens[p][pmatrix_index]) {
                ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] =
                        old_brlens[p][pmatrix_index];
                ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
                invalidateHigherCLVs(ann_network,
                        getTarget(ann_network.network, &ann_network.network.edges[i]), true,
                        visited);
            }
        }
    }
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

void assertBranchLengthsAndProbs(AnnotatedNetwork& ann_network, const std::vector<std::vector<double> >& old_brlens, const std::vector<std::vector<double> >& old_brprobs, double start_logl) {
    std::vector<std::vector<double> > act_brlens = extract_brlens(ann_network);
    std::vector<std::vector<double> > act_brprobs = extract_brprobs(ann_network);
    for (size_t i = 0; i < act_brlens.size(); ++i) {
        for (size_t j = 0; j < act_brlens[i].size(); ++j) {
            if (fabs(act_brlens[i][j] - old_brlens[i][j]) >= 1E-5) {
                std::cout << "wanted brlens:\n";
                for (size_t k = 0; k < ann_network.network.num_branches(); ++k) {
                    std::cout << "idx " << ann_network.network.edges[k].pmatrix_index << ": "
                            << old_brlens[i][k] << "\n";
                }
                std::cout << "\n";
                std::cout << "observed brlens:\n";
                for (size_t k = 0; k < ann_network.network.num_branches(); ++k) {
                    std::cout << "idx " << ann_network.network.edges[k].pmatrix_index << ": "
                            << act_brlens[i][k] << "\n";
                }
                std::cout << "\n";
            }
            assert(fabs(act_brlens[i][j] - old_brlens[i][j]) < 1E-5);
        }
    }
    for (size_t i = 0; i < act_brprobs.size(); ++i) {
        for (size_t j = 0; j < act_brprobs[i].size(); ++j) {
            if (fabs(act_brprobs[i][j] - old_brprobs[i][j]) >= 1E-5) {
                std::cout << "wanted brprobs:\n";
                for (size_t k = 0; k < ann_network.network.num_branches(); ++k) {
                    std::cout << "idx " << ann_network.network.edges[k].pmatrix_index << ": "
                            << old_brprobs[i][k] << "\n";
                }
                std::cout << "\n";
                std::cout << "observed brprobs:\n";
                for (size_t k = 0; k < ann_network.network.num_branches(); ++k) {
                    std::cout << "idx " << ann_network.network.edges[k].pmatrix_index << ": "
                            << act_brprobs[i][k] << "\n";
                }
                std::cout << "\n";
            }
            assert(fabs(act_brprobs[i][j] - old_brprobs[i][j]) < 1E-5);
        }
    }

    if (fabs(ann_network.raxml_treeinfo->loglh(true) - start_logl) >= 1E-5) {
        std::cout << "wanted: " << start_logl << "\n";
        std::cout << "observed: " << ann_network.raxml_treeinfo->loglh(true) << "\n";
        std::cout << "recomputed without incremental: "
                << ann_network.raxml_treeinfo->loglh(false) << "\n";
    }
}

template<typename T>
double hillClimbingStep(AnnotatedNetwork &ann_network, std::vector<T> candidates, double old_score, bool greedy=false, bool randomizeCandidates=false) {
    if (randomizeCandidates) {
        std::random_shuffle(candidates.begin(), candidates.end());
    }
    double start_logl = ann_network.raxml_treeinfo->loglh(true);
    std::vector<std::vector<double> > start_brlens = extract_brlens(ann_network);
    std::vector<std::vector<double> > start_brprobs = extract_brprobs(ann_network); // just for debug
    std::string before = exportDebugInfo(ann_network.network);

    size_t best_idx = candidates.size();
    double best_score = old_score;
    std::vector<std::vector<double> > best_brlens = start_brlens;

    for (size_t i = 0; i < candidates.size(); ++i) {
        performMove(ann_network, candidates[i]);
        // TODO: Also do brlen optimization locally around the move
        //std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, candidates[i]);
        //optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);

        double new_logl = ann_network.raxml_treeinfo->loglh(true);
        double new_score = bic(ann_network, new_logl);
        bool foundBetterScore = false;
        if (new_score < best_score) {
            best_score = new_score;
            best_idx = i;
            //best_brlens = extract_brlens(ann_network);
            foundBetterScore = true;
        }
        undoMove(ann_network, candidates[i]);
        assert(exportDebugInfo(ann_network.network) == before);
        //apply_brlens(ann_network, start_brlens);
        // TODO: Figure out why these assertions fail, fix it, then add brlen opt
        assertBranchLengthsAndProbs(ann_network, start_brlens, start_brprobs, start_logl);
        assert(ann_network.raxml_treeinfo->loglh(true) == start_logl);
        if (greedy && foundBetterScore) {
            break;
        }
    }

    if (best_idx < candidates.size()) {
        performMove(ann_network, candidates[best_idx]);
        //apply_brlens(ann_network, best_brlens);
    }
    return best_score;
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, MoveType type) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double old_bic = bic(ann_network, old_logl);
    //std::cout << "start_logl: " << old_logl <<", start_bic: " << old_bic << "\n";

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
    } while (new_score < old_bic);
    return ann_network.raxml_treeinfo->loglh(true);
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network) {
    std::vector<MoveType> types = { MoveType::ArcRemovalMove, MoveType::RNNIMove,
            MoveType::RSPRMove, MoveType::ArcInsertionMove };
    unsigned int type_idx = 0;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double new_logl = old_logl;
    double old_bic = bic(ann_network, old_logl);
    double new_score = old_bic;
    unsigned int moves_cnt = 0;
    do {
        //std::cout << toExtendedNewick(ann_network.network) << "\n";
        //std::cout << "Using move type: " << toString(types[type_idx]) << "\n";
        //std::cout << toExtendedNewick(ann_network.network) << "\n";
        //std::cout << exportDebugInfo(ann_network.network) << "\n";
        new_logl = greedyHillClimbingTopology(ann_network, types[type_idx]);
        new_score = bic(ann_network, new_logl);
        type_idx = (type_idx + 1) % types.size();
        moves_cnt++;
        if (ann_network.network.num_reticulations() == 0
                && (types[type_idx] == MoveType::DeltaMinusMove
                        || types[type_idx] == MoveType::ArcRemovalMove)) {
            type_idx = (type_idx + 1) % types.size();
            moves_cnt++;
        }

        if (new_score < old_bic) {
            std::cout << "Improved bic from " << old_bic << " to " << new_score << "\n";
            old_bic = new_score;
            moves_cnt = 0;
        }
    } while ((new_score < old_bic) || (moves_cnt < types.size()));
    return new_logl;
}

double simulatedAnnealingTopology(AnnotatedNetwork &ann_network, MoveType type) {
    throw std::runtime_error("Not implemented yet");
}

double simulatedAnnealingTopology(AnnotatedNetwork &ann_network) {
    throw std::runtime_error("Not implemented yet");
}

}
