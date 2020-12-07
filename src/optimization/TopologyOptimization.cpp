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
            assert(ann_network.network.edges_by_index[pmatrix_index] == &ann_network.network.edges[i]);
            if (n_partitions == 1) {
                assert(ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] == ann_network.network.edges_by_index[pmatrix_index]->length);
            }
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
                if (n_partitions == 1) {
                    ann_network.network.edges_by_index[pmatrix_index]->length = old_brlens[p][pmatrix_index];
                }
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
    bool unlinked_mode = (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t multiplier = unlinked_mode ? ann_network.fake_treeinfo->partition_count : 1;
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

template<typename T>
double hillClimbingStep(AnnotatedNetwork &ann_network, std::vector<T> candidates, double old_score, bool greedy=true, bool randomizeCandidates=false, bool brlenopt_inside=true) {
    if (randomizeCandidates) {
        std::random_shuffle(candidates.begin(), candidates.end());
    }
    int max_iters = 1;
    int radius = 1;
    double start_logl = ann_network.raxml_treeinfo->loglh(true);
    std::vector<std::vector<double> > start_brlens;
    if (brlenopt_inside) {
        start_brlens = extract_brlens(ann_network);
    }

    size_t best_idx = candidates.size();
    double best_score = old_score;
    std::vector<std::vector<double> > best_brlens = start_brlens;

    for (size_t i = 0; i < candidates.size(); ++i) {

        std::cout << "Extensive BIC info before applying current " << toString(candidates[i].moveType) << " move " << i+1 << "/ " << candidates.size() << ":\n";
        printExtensiveBICInfo(ann_network);

        //std::cout << " " << toString(candidates[i].moveType) << " " << i+1 << "/ " << candidates.size() << "\n";
        performMove(ann_network, candidates[i]);
        if (brlenopt_inside) { // Do brlen optimization locally around the move
            std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, candidates[i]);
            optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
        }

        std::cout << "Extensive BIC info after applying current " << toString(candidates[i].moveType) << " move " << i+1 << "/ " << candidates.size() << ":\n";
        printExtensiveBICInfo(ann_network);

        double new_logl = ann_network.raxml_treeinfo->loglh(true);
        double new_score = bic(ann_network, new_logl);
        bool foundBetterScore = false;
        if (new_score < best_score) {
            best_score = new_score;
            best_idx = i;
            if (brlenopt_inside) {
                best_brlens = extract_brlens(ann_network);
            }
            foundBetterScore = true;
        }
        undoMove(ann_network, candidates[i]);
        if (brlenopt_inside) {
            apply_brlens(ann_network, start_brlens);
        }
        if (fabs(ann_network.raxml_treeinfo->loglh(true) - start_logl) >= ann_network.options.lh_epsilon) {
            std::cout << "new value: " << ann_network.raxml_treeinfo->loglh(true) << "\n";
            std::cout << "old value: " << start_logl << "/n";
        }
        assert(fabs(ann_network.raxml_treeinfo->loglh(true) - start_logl) < ann_network.options.lh_epsilon);
        if (greedy && foundBetterScore) {
            break;
        }
    }

    if (best_idx < candidates.size()) {
        performMove(ann_network, candidates[best_idx]);
        if (brlenopt_inside) {
            apply_brlens(ann_network, best_brlens);
        } else {
            std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, candidates[best_idx]);
            optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
        }
        // just for debug, doing reticulation opt, full global brlen opt and model opt:
        //netrax::computeLoglikelihood(ann_network, 0, 1, false);
        //ann_network.raxml_treeinfo->optimize_branches(ann_network.options.lh_epsilon, 1);
        //ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);

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

        std::cout << "  param_count: " << get_param_count(ann_network) << ", sample_size:" << get_sample_size << "\n";
        std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
        std::cout << toExtendedNewick(ann_network.network) << "\n";
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
    } while (old_bic - new_score > ann_network.options.lh_epsilon);
    return ann_network.raxml_treeinfo->loglh(true);
}

double greedyHillClimbingTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types) {
    unsigned int type_idx = 0;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double new_logl = old_logl;
    double old_bic = bic(ann_network, old_logl);
    double new_score = old_bic;
    unsigned int moves_cnt = 0;
    do {
        if (ann_network.network.num_reticulations() == 0
              && (types[type_idx] == MoveType::DeltaMinusMove || types[type_idx] == MoveType::ArcRemovalMove)) {
            type_idx = (type_idx + 1) % types.size();
            moves_cnt++;
        }
        //std::cout << "Using move type: " << toString(types[type_idx]) << "\n";
        new_logl = greedyHillClimbingTopology(ann_network, types[type_idx]);
        new_score = bic(ann_network, new_logl);
        type_idx = (type_idx + 1) % types.size();
        moves_cnt++;

        if (old_bic - new_score > ann_network.options.lh_epsilon) {
            //std::cout << "Improved bic from " << old_bic << " to " << new_score << "\n";
            old_bic = new_score;
            moves_cnt = 0;
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
