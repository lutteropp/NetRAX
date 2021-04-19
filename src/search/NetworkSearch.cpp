#include "NetworkSearch.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <limits>
#include <omp.h>

#include "../likelihood/mpreal.h"
#include "../graph/AnnotatedNetwork.hpp"
#include "../io/NetworkIO.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../optimization/ModelOptimization.hpp"
#include "../optimization/ReticulationOptimization.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/Moves.hpp"
#include "../optimization/MoveType.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../optimization/NetworkState.hpp"

//#undef _NETRAX_OPENMP

namespace netrax {

struct ScoreImprovementResult {
    bool local_improved = false;
    bool global_improved = false;
};

bool can_write(){
    return (ParallelContext::master_rank() && ParallelContext::master_thread());
}

bool logl_stays_same(AnnotatedNetwork& ann_network) {
    if (can_write()) {
        std::cout << "displayed trees before:\n";
        for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
            size_t n_trees = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees;
            for (size_t j = 0; j < n_trees; ++j) {
                DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[j];
                std::cout << "logl: " << tree.treeLoglData.tree_partition_logl[i] << ", logprob: " << tree.treeLoglData.tree_logprob << "\n";
            }
        }
    }
    double incremental = netrax::computeLoglikelihood(ann_network, 1, 1);
    if (can_write()) {
        std::cout << "displayed trees in between:\n";
        for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
            size_t n_trees = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees;
            for (size_t j = 0; j < n_trees; ++j) {
                DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[j];
                std::cout << "logl: " << tree.treeLoglData.tree_partition_logl[i] << ", logprob: " << tree.treeLoglData.tree_logprob << "\n";
            }
        }
    }
    double normal = netrax::computeLoglikelihood(ann_network, 0, 1);
    if (can_write()) {
        std::cout << "displayed trees after:\n";
        for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
            size_t n_trees = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees;
            for (size_t j = 0; j < n_trees; ++j) {
                DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[j];
                std::cout << "logl: " << tree.treeLoglData.tree_partition_logl[i] << ", logprob: " << tree.treeLoglData.tree_logprob << "\n";
            }
        }
    }

    if (can_write()) {
        std::cout << "incremental: " << incremental << "\n";
        std::cout << "normal: " << normal << "\n";
    }
    return (incremental == normal);
}

void optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt, bool silent) {
    assert(logl_stays_same(ann_network));
    bool gotBetter = true;
    while (gotBetter) {
        gotBetter = false;
        double score_before = scoreNetwork(ann_network);
        optimizeModel(ann_network, silent);
        optimizeBranches(ann_network, silent);
        optimizeReticulationProbs(ann_network, silent);
        double score_after = scoreNetwork(ann_network);

        if (score_after < score_before && extremeOpt) {
            gotBetter = true;
        }
    }

    assert(logl_stays_same(ann_network));
}

void printDisplayedTrees(AnnotatedNetwork& ann_network) {
    if (can_write()) {
        std::vector<std::pair<std::string, double>> displayed_trees;
        if (ann_network.network.num_reticulations() == 0) {
            std::string newick = netrax::toExtendedNewick(ann_network);
            displayed_trees.emplace_back(std::make_pair(newick, 1.0));
        } else {
            size_t n_trees = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees;
            for (size_t j = 0; j < n_trees; ++j) {
                DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[j];
                pll_utree_t* utree = netrax::displayed_tree_to_utree(ann_network.network, tree.treeLoglData.reticulationChoices.configs[0]);
                double prob = std::exp(tree.treeLoglData.tree_logprob);
                Network displayedNetwork = netrax::convertUtreeToNetwork(*utree, 0);
                std::string newick = netrax::toExtendedNewick(displayedNetwork);
                pll_utree_destroy(utree, nullptr);
                displayed_trees.emplace_back(std::make_pair(newick, prob));
            }
        }
        std::cout << "Number of displayed trees: " << displayed_trees.size() << "\n";
        std::cout << "Displayed trees Newick strings:\n";
        for (const auto& entry : displayed_trees) {
            std::cout << entry.first << "\n";
        }
        std::cout << "Displayed trees probabilities:\n";
        for (const auto& entry : displayed_trees) {
            std::cout << entry.second << "\n";
        }
    }
}

struct BestNetworkData {
    size_t best_n_reticulations = 0;
    std::vector<double> logl;
    std::vector<double> bic;
    std::vector<std::string> newick;
    BestNetworkData(size_t max_reticulations) {
        logl = std::vector<double>(max_reticulations + 1, -std::numeric_limits<double>::infinity());
        bic.resize(max_reticulations + 1, std::numeric_limits<double>::infinity());
        newick.resize(max_reticulations + 1, "");
    }
};

bool hasBadReticulation(AnnotatedNetwork& ann_network) {
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        if ((1.0 - ann_network.reticulation_probs[i] < 0.001) || (ann_network.reticulation_probs[i] < 0.001)) {
            return true;
        }
    }
    return false;
}

ScoreImprovementResult check_score_improvement(AnnotatedNetwork& ann_network, double* local_best, BestNetworkData* bestNetworkData, bool silent = true) {
    bool local_improved = false;
    bool global_improved = false;
    
    double new_score = scoreNetwork(ann_network);
    double score_diff = bestNetworkData->bic[ann_network.network.num_reticulations()] - new_score;
    if (score_diff > 0) {
        double old_global_best = bestNetworkData->bic[bestNetworkData->best_n_reticulations];
        bestNetworkData->bic[ann_network.network.num_reticulations()] = new_score;
        bestNetworkData->logl[ann_network.network.num_reticulations()] = computeLoglikelihood(ann_network, 1, 1);
        bestNetworkData->newick[ann_network.network.num_reticulations()] = toExtendedNewick(ann_network);
        
        local_improved = true;

        if (new_score < old_global_best) {
            if (hasBadReticulation(ann_network)) {
                if (can_write()) {
                    if (!silent) std::cout << "Network contains BAD RETICULATIONS. Not updating the global best found network and score.\n";
                }
            } else {
                bestNetworkData->best_n_reticulations = ann_network.network.num_reticulations();
                global_improved = true;
                //std::cout << "OLD GLOBAL BEST SCORE WAS: " << old_global_best << "\n";
                if (can_write()) {
                    std::cout << "IMPROVED GLOBAL BEST SCORE FOUND SO FAR: " << new_score << "\n";
                    writeNetwork(ann_network, ann_network.options.output_file);
                    if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
                    if (!silent) std::cout << "Better network written to " << ann_network.options.output_file << "\n";
                }
                //printDisplayedTrees(ann_network);
            }
            *local_best = new_score;
        } else if (new_score < *local_best) {
            //std::cout << "SCORE DIFF: " << score_diff << "\n";
            //std::cout << "OLD LOCAL BEST SCORE WAS: " << *local_best << "\n";
            *local_best = new_score;
            //std::cout << "IMPROVED LOCAL BEST SCORE FOUND SO FAR: " << new_score << "\n\n";
        }
    }
    return ScoreImprovementResult{local_improved, global_improved};
}

double getWorstReticulationScore(AnnotatedNetwork& ann_network) {
    double worst = 1.0;
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        double score = fabs(0.5 - ann_network.reticulation_probs[i]);
        score = 1.0 - (2.0 * score); // 1.0 is best result, 0 is worst result

        worst = std::min(worst, score);
    }
    return worst; // 1.0 is best result, 0 is worst result
}

template <typename T>
struct ScoreItem {
    T move;
    double worstScore;
    double bicScore;
};

bool isComplexityChangingMove(const MoveType& moveType) {
    return (moveType == MoveType::ArcRemovalMove || moveType == MoveType::ArcInsertionMove || moveType == MoveType::DeltaMinusMove || moveType == MoveType::DeltaPlusMove);
}

template <typename T>
void printCandidates(std::vector<T>& candidates) {
    if (can_write()) {
        std::cout << "The candidates are:\n";
        for (size_t i = 0; i < candidates.size(); ++i) {
            std::cout << toString(candidates[i]) << "\n";
        }
        std::cout << "End of candidates.\n";
    }
}

bool needsRecompute(AnnotatedNetwork& ann_network, const ArcRemovalMove& move) {
    return (ann_network.network.reticulation_nodes[ann_network.network.num_reticulations() - 1]->clv_index != move.v_clv_index);
}
bool needsRecompute(AnnotatedNetwork& ann_network, const ArcInsertionMove& move) {
    return false;
}
bool needsRecompute(AnnotatedNetwork& ann_network, const RSPRMove& move) {
    return false;
}
bool needsRecompute(AnnotatedNetwork& ann_network, const RNNIMove& move) {
    return false;
}
bool needsRecompute(AnnotatedNetwork& ann_network, GeneralMove* move) {
    assert(move);
    return (move->moveType == MoveType::ArcRemovalMove) && (ann_network.network.reticulation_nodes[ann_network.network.num_reticulations() - 1]->clv_index != ((ArcRemovalMove*) move)->v_clv_index);
}

template <typename T>
void prefilterCandidates(AnnotatedNetwork& ann_network_orig, std::vector<T>& candidates, std::vector<AnnotatedNetwork>& ann_network_thread, bool silent = true) {
    if (candidates.empty()) {
        return;
    }

    double brlen_smooth_factor = 0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int max_iters_outside = max_iters;
    int radius = 1;
    double old_bic = scoreNetwork(ann_network_orig);

    double best_bic = std::numeric_limits<double>::infinity();

    NetworkState oldState = extract_network_state(ann_network_orig);

    std::vector<ScoreItem<T> > scores(candidates.size());

    bool stop = false;

    #ifdef _NETRAX_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (stop) {
            continue;
        }
        #ifdef _NETRAX_OPENMP
        AnnotatedNetwork& ann_network = ann_network_thread[omp_get_thread_num()];
        #else
        AnnotatedNetwork& ann_network = ann_network_orig;
        #endif
        apply_network_state(ann_network, oldState);
        T move(candidates[i]);
        bool recompute_from_scratch = needsRecompute(ann_network, move);

        assert(checkSanity(ann_network, move));

        performMove(ann_network, move);
        if (recompute_from_scratch) {
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        optimizeReticulationProbs(ann_network);
        
        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        
        optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates, true);
        /*
        if (move->moveType == MoveType::ArcInsertionMove || move->moveType == MoveType::DeltaPlusMove) {
            optimize_branches(ann_network, max_iters, 1, radius, brlen_opt_candidates, false);
        } else {
            optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates, true);
        }*/

        double bicScore = scoreNetwork(ann_network);
        double worstScore = getWorstReticulationScore(ann_network);

        scores[i] = ScoreItem<T>{candidates[i], worstScore, bicScore};

        for (size_t j = 0; j < ann_network.network.num_nodes(); ++j) {
            assert(ann_network.network.nodes_by_index[j]->clv_index == j);
        }

        if (bicScore < best_bic) {
            #ifdef _NETRAX_OPENMP
            #pragma omp critical
            {
                if (bicScore < best_bic) {
                    best_bic = bicScore;
                }
            }
            #else
            best_bic = bicScore;
            #endif
        }

        if (ann_network.options.use_extreme_greedy) {
            if (bicScore < old_bic && !stop) {
                #ifdef _NETRAX_OPENMP
                #pragma omp critical
                {
                    if (bicScore < old_bic && !stop) {
                        candidates[0] = candidates[i];
                        stop = true;
                    }
                }
                #else
                candidates[0] = candidates[i];
                stop = true;
                #endif
            }
        }
    }

    if (stop) {
        candidates.resize(1);
        return;
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        if (lhs.bicScore == rhs.bicScore) {
            return lhs.worstScore > rhs.worstScore;
        }
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    double cutoff_bic = std::min(best_bic, old_bic);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (can_write()) {
            if (!silent) std::cout << "prefiltered candidate " << i + 1 << "/" << candidates.size() << " has worst score " << scores[i].worstScore << ", BIC: " << scores[i].bicScore << "\n";
        }
        if (scores[i].bicScore <= cutoff_bic) {
            candidates[newSize] = scores[i].move;
            newSize++;
        }
    }
    if (can_write()) {
        if (!silent) std::cout << "New size after prefiltering: " << newSize << " vs. " << candidates.size() << "\n";
    }

    candidates.resize(newSize);

    #ifndef _NETRAX_OPENMP
    apply_network_state(ann_network_orig, oldState);
    #endif

    for (size_t i = 0; i < candidates.size(); ++i) {
        assert(checkSanity(ann_network_orig, candidates[i]));
    }
}

template <typename T>
bool rankCandidates(AnnotatedNetwork& ann_network_orig, std::vector<T> candidates, NetworkState* state, std::vector<AnnotatedNetwork>& ann_network_thread, bool silent = true) {
    if (candidates.empty()) {
        return false;
    }
    if (!ann_network_orig.options.no_prefiltering) {
        prefilterCandidates(ann_network_orig, candidates, ann_network_thread, true);
    }

    if (can_write()) {
        if (!silent) std::cout << "MoveType: " << toString(candidates[0].moveType) << "\n";
    }

    double brlen_smooth_factor = 1.0;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int max_iters_outside = max_iters;
    int radius = 1;

    double old_bic = scoreNetwork(ann_network_orig);
    double best_bic = old_bic;
    bool found_better = false;

    NetworkState oldState = extract_network_state(ann_network_orig);

    std::vector<ScoreItem<T> > scores(candidates.size());

    bool stop = false;

    #ifdef _NETRAX_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (stop) {
            continue;
        }
        #ifdef _NETRAX_OPENMP
        AnnotatedNetwork& ann_network = ann_network_thread[omp_get_thread_num()];
        #else
        AnnotatedNetwork& ann_network = ann_network_orig;
        #endif
        apply_network_state(ann_network, oldState);
        T move(candidates[i]);
        bool recompute_from_scratch = needsRecompute(ann_network, move);

        assert(checkSanity(ann_network, move));

        performMove(ann_network, move);
        if (recompute_from_scratch) {
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
        optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates);
        optimizeReticulationProbs(ann_network);

        double worstScore = getWorstReticulationScore(ann_network);
        double bicScore = scoreNetwork(ann_network);

        if (bicScore < best_bic && !stop) {
            #ifdef _NETRAX_OPENMP
            #pragma omp critical
            {
                if (bicScore < best_bic && !stop) {
                    best_bic = bicScore;
                    if (found_better) {
                        extract_network_state(ann_network, *state);
                    } else {
                        *state = extract_network_state(ann_network);
                    }
                    found_better = true;
                }
            }
            #else
            best_bic = bicScore;
            if (found_better) {
                extract_network_state(ann_network, *state);
            } else {
                *state = extract_network_state(ann_network);
            }
            found_better = true;
            #endif
        }

        if (ann_network.options.use_extreme_greedy) {
            if (bicScore < old_bic && !stop) {
                #ifdef _NETRAX_OPENMP
                #pragma omp critical
                {
                    if (bicScore < old_bic && !stop) {
                        candidates[0] = candidates[i];
                        stop = true;
                    }
                }
                #else
                candidates[0] = candidates[i];
                stop = true;
                #endif
            }
        }

        scores[i] = ScoreItem<T>{candidates[i], worstScore, bicScore};
    }

    if (stop) {
        candidates.resize(1);
        return found_better;
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        if (lhs.bicScore == rhs.bicScore) {
            return lhs.worstScore > rhs.worstScore;
        }
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (can_write()) {
            if (!silent) std::cout << "candidate " << i + 1 << "/" << candidates.size() << " has worst score " << scores[i].worstScore << ", BIC: " << scores[i].bicScore << "\n";
        }
        if (scores[i].bicScore < old_bic) {
            candidates[newSize] = scores[i].move;
            newSize++;
        }
    }

    candidates.resize(newSize);

    #ifndef _NETRAX_OPENMP
        apply_network_state(ann_network_orig, oldState);
    #endif

    return found_better;
}

template <typename T>
double applyBestCandidate(AnnotatedNetwork& ann_network, std::vector<T> candidates, double* best_score, BestNetworkData* bestNetworkData, std::vector<AnnotatedNetwork>& ann_network_thread, bool silent = true) {
    NetworkState state;
    bool found_better_state = rankCandidates(ann_network, candidates, &state, ann_network_thread, true);

    if (found_better_state) {
        apply_network_state(ann_network, state);

        optimizeAllNonTopology(ann_network);
        double logl = computeLoglikelihood(ann_network);
        double bic_score = bic(ann_network, logl);
        double aic_score = aic(ann_network, logl);
        double aicc_score = aicc(ann_network, logl);

        if (can_write()) {
            if (!silent) std::cout << " Took " << toString(candidates[0].moveType) << "\n";
            if (!silent) std::cout << "  Logl: " << logl << ", BIC: " << bic_score << ", AIC: " << aic_score << ", AICc: " << aicc_score <<  "\n";
            if (!silent) std::cout << "  param_count: " << get_param_count(ann_network) << ", sample_size:" << get_sample_size(ann_network) << "\n";
            if (!silent) std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
            if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
        }
        ann_network.stats.moves_taken[candidates[0].moveType]++;

        check_score_improvement(ann_network, best_score, bestNetworkData);
    }

    return computeLoglikelihood(ann_network);
}

template <typename T>
bool simanneal_step(AnnotatedNetwork& ann_network, double t, std::vector<T> neighbors, const NetworkState& oldState, std::unordered_set<double>& seen_bics, std::vector<AnnotatedNetwork>& ann_network_thread, bool silent = true) {
    if (neighbors.empty() || t <= 0) {
        return false;
    }

    if (!ann_network.options.no_prefiltering) {
        prefilterCandidates(ann_network, neighbors, ann_network_thread, true);
    }

    //if (!silent) std::cout << "MoveType: " << toString(neighbors[0].moveType) << "\n";
    if (can_write()) {
        if (!silent) std::cout << "t: " << t << "\n";
    }

    double brlen_smooth_factor = 0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int max_iters_outside = max_iters;
    int radius = 1;

    double old_bic = scoreNetwork(ann_network);

    for (size_t i = 0; i < neighbors.size(); ++i) {
        T move(neighbors[i]);
        assert(checkSanity(ann_network, move));
        bool recompute_from_scratch = needsRecompute(ann_network, move);
        performMove(ann_network, move);
        if (recompute_from_scratch) {
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
        optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates);
        optimizeReticulationProbs(ann_network);
        
        double bicScore = scoreNetwork(ann_network);

        if (bicScore < old_bic) {
            if (can_write()) {
                if (!silent) std::cout << " Took " << toString(move.moveType) << "\n";
                if (!silent) std::cout << "  Logl: " << computeLoglikelihood(ann_network) << ", BIC: " << scoreNetwork(ann_network) << "\n";
                if (!silent) std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
                if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
            }
            return true;
        }

        if (seen_bics.count(bicScore) == 0) {
            seen_bics.emplace(bicScore);
            double acceptance_ratio = exp(-((bicScore - old_bic) / t)); // I took this one from: https://de.wikipedia.org/wiki/Simulated_Annealing
            double x = std::uniform_real_distribution<double>(0,1)(ann_network.rng);
            if (x <= acceptance_ratio) {
                if (can_write()) {
                    if (!silent) std::cout << " Took " << toString(move.moveType) << "\n";
                    if (!silent) std::cout << "  Logl: " << computeLoglikelihood(ann_network) << ", BIC: " << scoreNetwork(ann_network) << "\n";
                    if (!silent) std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
                }
                //if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
                return true;
            }
        }
        apply_network_state(ann_network, oldState);
        assert(checkSanity(ann_network, neighbors[i]));
    }

    return false;
}

double update_temperature(double t) {
    return t*0.95; // TODO: Better temperature update ? I took this one from: https://de.mathworks.com/help/gads/how-simulated-annealing-works.html
}

double simanneal(AnnotatedNetwork& ann_network, double t_start, MoveType type, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, BestNetworkData* bestNetworkData, std::vector<AnnotatedNetwork>& ann_network_thread, bool silent = false) {
    double start_bic = scoreNetwork(ann_network);
    double best_bic = start_bic;
    extract_network_state(ann_network, best_state_to_reuse);
    extract_network_state(ann_network, start_state_to_reuse);
    double t = t_start;
    bool network_changed = true;
    std::unordered_set<double> seen_bics;
    while (network_changed) {
        network_changed = false;
        extract_network_state(ann_network, start_state_to_reuse);

        switch (type) {
        case MoveType::RNNIMove:
            network_changed = simanneal_step(ann_network, t, possibleRNNIMoves(ann_network), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::RSPRMove:
            network_changed = simanneal_step(ann_network, t, possibleRSPRMoves(ann_network, ann_network.options.less_moves), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::RSPR1Move:
            network_changed = simanneal_step(ann_network, t, possibleRSPR1Moves(ann_network), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::HeadMove:
            network_changed = simanneal_step(ann_network, t, possibleHeadMoves(ann_network, true), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::TailMove:
            network_changed = simanneal_step(ann_network, t, possibleTailMoves(ann_network, true), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::ArcInsertionMove:
            network_changed = simanneal_step(ann_network, t, possibleArcInsertionMoves(ann_network, true), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::DeltaPlusMove:
            network_changed = simanneal_step(ann_network, t, possibleDeltaPlusMoves(ann_network), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::ArcRemovalMove:
            network_changed = simanneal_step(ann_network, t, possibleArcRemovalMoves(ann_network), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        case MoveType::DeltaMinusMove:
            network_changed = simanneal_step(ann_network, t, possibleDeltaMinusMoves(ann_network), start_state_to_reuse, seen_bics, ann_network_thread);
            break;
        default:
            throw std::runtime_error("Invalid move type");
        }

        if (network_changed) {
            double act_bic = scoreNetwork(ann_network);
            if (act_bic < best_bic) {
                optimizeAllNonTopology(ann_network, true);
                check_score_improvement(ann_network, &best_bic, bestNetworkData);
                extract_network_state(ann_network, best_state_to_reuse);
            }
        }

        t = update_temperature(t);
    }

    apply_network_state(ann_network, best_state_to_reuse);
    return computeLoglikelihood(ann_network);
}

bool isArcInsertion(const MoveType& type) {
    return (type == MoveType::ArcInsertionMove || type == MoveType::DeltaPlusMove);
}

bool isArcRemoval(const MoveType& type) {
    return (type == MoveType::ArcRemovalMove || type == MoveType::DeltaMinusMove);
}

double optimizeEverythingRun(AnnotatedNetwork& ann_network, const std::vector<MoveType>& typesBySpeed, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, const std::chrono::high_resolution_clock::time_point& start_time, BestNetworkData* bestNetworkData, std::vector<AnnotatedNetwork>& ann_network_thread) {
    unsigned int type_idx = 0;
    unsigned int max_seconds = ann_network.options.timeout;
    double best_score = scoreNetwork(ann_network);
    do {
        while (ann_network.network.num_reticulations() == 0 && isArcRemoval(typesBySpeed[type_idx])) {
            type_idx++;
            if (type_idx >= typesBySpeed.size()) {
                break;
            }
        }
        if (type_idx >= typesBySpeed.size()) {
            break;
        }
        while (ann_network.network.num_reticulations() == ann_network.options.max_reticulations && isArcInsertion(typesBySpeed[type_idx])) {
            type_idx++;
            if (type_idx >= typesBySpeed.size()) {
                break;
            }
        }
        if (type_idx >= typesBySpeed.size()) {
            break;
        }
        double old_score = scoreNetwork(ann_network);

        if (ann_network.options.sim_anneal && !isComplexityChangingMove(typesBySpeed[type_idx])) {
            simanneal(ann_network, ann_network.options.start_temperature, typesBySpeed[type_idx], start_state_to_reuse, best_state_to_reuse, bestNetworkData, ann_network_thread);
        } else {
            switch (typesBySpeed[type_idx]) {
            case MoveType::RNNIMove:
                applyBestCandidate(ann_network, possibleRNNIMoves(ann_network), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::RSPRMove:
                applyBestCandidate(ann_network, possibleRSPRMoves(ann_network, ann_network.options.less_moves), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::RSPR1Move:
                applyBestCandidate(ann_network, possibleRSPR1Moves(ann_network), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::HeadMove:
                applyBestCandidate(ann_network, possibleHeadMoves(ann_network, true), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::TailMove:
                applyBestCandidate(ann_network, possibleTailMoves(ann_network, true), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::ArcInsertionMove:
                applyBestCandidate(ann_network, possibleArcInsertionMoves(ann_network, true), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::DeltaPlusMove:
                applyBestCandidate(ann_network, possibleDeltaPlusMoves(ann_network), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::ArcRemovalMove:
                applyBestCandidate(ann_network, possibleArcRemovalMoves(ann_network), &best_score, bestNetworkData, ann_network_thread);
                break;
            case MoveType::DeltaMinusMove:
                applyBestCandidate(ann_network, possibleDeltaMinusMoves(ann_network), &best_score, bestNetworkData, ann_network_thread);
                break;
            default:
                throw std::runtime_error("Invalid move type");
            }
        }

        double new_score = scoreNetwork(ann_network);
        if (new_score < old_score) { // score got better
            new_score = scoreNetwork(ann_network);
            best_score = new_score;

            type_idx = 0; // go back to fastest move type        
        } else { // try next-slower move type
            type_idx++;
        }
        assert(new_score <= old_score);

        if (max_seconds != 0) {
            auto act_time = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>( act_time - start_time ).count() >= max_seconds) {
                break;
            }
        }
    } while (type_idx < typesBySpeed.size());

    optimizeAllNonTopology(ann_network, true);
    best_score = scoreNetwork(ann_network);

    return best_score;
}

void scrambleNetwork(AnnotatedNetwork& ann_network, MoveType type, size_t scramble_cnt) {
    // perform scramble_cnt moves of the specified move type on the network
    ArcInsertionMove insertionMove;
    ArcRemovalMove removalMove;
    RNNIMove rnniMove;
    RSPRMove rsprMove;
    for (size_t i = 0; i < scramble_cnt; ++i) {
        switch (type) {
        case MoveType::ArcInsertionMove:
            insertionMove = randomArcInsertionMove(ann_network);
            performMove(ann_network, insertionMove);
            break;
        case MoveType::ArcRemovalMove:
            removalMove = randomArcRemovalMove(ann_network);
            performMove(ann_network, removalMove);
            break;
        case MoveType::DeltaMinusMove:
            removalMove = randomDeltaMinusMove(ann_network);
            performMove(ann_network, removalMove);
            break;
        case MoveType::DeltaPlusMove:
            insertionMove = randomDeltaPlusMove(ann_network);
            performMove(ann_network, insertionMove);
            break;
        case MoveType::HeadMove:
            rsprMove = randomHeadMove(ann_network);
            performMove(ann_network, rsprMove);
            break;
        case MoveType::RNNIMove:
            rnniMove = randomRNNIMove(ann_network);
            performMove(ann_network, rnniMove);
            break;
        case MoveType::RSPR1Move:
            rsprMove = randomRSPR1Move(ann_network);
            performMove(ann_network, rsprMove);
            break;
        case MoveType::RSPRMove:
            rsprMove = randomRSPRMove(ann_network);
            performMove(ann_network, rsprMove);
            break;
        case MoveType::TailMove:
            rsprMove = randomTailMove(ann_network);
            performMove(ann_network, rsprMove);
            break;
        default:
            throw std::runtime_error("Unrecognized move type");
        }
    }
    optimizeAllNonTopology(ann_network);
}

void wavesearch(AnnotatedNetwork& ann_network, BestNetworkData* bestNetworkData, const std::vector<MoveType>& typesBySpeed, std::vector<AnnotatedNetwork>& ann_network_thread, bool silent = false) {
    NetworkState start_state_to_reuse = extract_network_state(ann_network, false);
    NetworkState best_state_to_reuse = extract_network_state(ann_network, false);
    auto start_time = std::chrono::high_resolution_clock::now();
    double best_score = std::numeric_limits<double>::infinity();
    ScoreImprovementResult score_improvement;

    //std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";

    optimizeAllNonTopology(ann_network, true);
    //std::cout << "Initial network after modelopt+brlenopt+reticulation opt is:\n" << toExtendedNewick(ann_network) << "\n\n";
    std::string best_network = toExtendedNewick(ann_network);
    score_improvement = check_score_improvement(ann_network, &best_score, bestNetworkData);

    optimizeEverythingRun(ann_network, typesBySpeed, start_state_to_reuse, best_state_to_reuse, start_time, bestNetworkData, ann_network_thread);
    score_improvement = check_score_improvement(ann_network, &best_score, bestNetworkData);

    if (ann_network.options.scrambling > 0) {
        if (can_write()) {
            std::cout << " Starting scrambling phase...\n";
        }
        unsigned int tries = 0;
        NetworkState bestState = extract_network_state(ann_network);
        if (can_write()) {
            if (!silent) std::cout << " Network before scrambling has BIC Score: " << scoreNetwork(ann_network) << "\n";
        }
        while (tries < ann_network.options.scrambling) {
            apply_network_state(ann_network, bestState);
            scrambleNetwork(ann_network, MoveType::RSPRMove, 2);
            optimizeEverythingRun(ann_network, typesBySpeed, start_state_to_reuse, best_state_to_reuse, start_time, bestNetworkData, ann_network_thread);
            if (can_write()) {
                if (!silent) std::cout << " scrambling BIC: " << scoreNetwork(ann_network) << "\n";
            }
            score_improvement = check_score_improvement(ann_network, &best_score, bestNetworkData);
            if (score_improvement.local_improved) {
                extract_network_state(ann_network, bestState);
                tries = 0;
            } else {
                tries++;
            }
        }
        apply_network_state(ann_network, bestState);
    }
}

void run_single_start_waves(const NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng) {
    /* non-master ranks load starting trees from a file */
    ParallelContext::global_mpi_barrier();
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions, instance);
    init_annotated_network(ann_network, rng);
    #ifdef _NETRAX_OPENMP
    std::vector<AnnotatedNetwork> ann_network_thread(omp_get_max_threads(), AnnotatedNetwork(ann_network));
    #else
    std::vector<AnnotatedNetwork> ann_network_thread;
    #endif
    BestNetworkData bestNetworkData(ann_network.options.max_reticulations);
    wavesearch(ann_network, &bestNetworkData, typesBySpeed, ann_network_thread);

    if (can_write()) {
        std::cout << "Statistics on which moves were taken:\n";
        for (const MoveType& type : typesBySpeed) {
            std::cout << toString(type) << ": " << ann_network.stats.moves_taken[type] << "\n";
        }
        std::cout << "Best inferred network has " << bestNetworkData.best_n_reticulations << " reticulations, logl = " << bestNetworkData.logl[bestNetworkData.best_n_reticulations] << ", bic = " << bestNetworkData.bic[bestNetworkData.best_n_reticulations] << "\n";
        std::cout << "Best inferred network is: \n";
        std::cout << bestNetworkData.newick[bestNetworkData.best_n_reticulations] << "\n";

        std::cout << "n_reticulations, logl, bic, newick\n";
        for (size_t i = 0; i < bestNetworkData.bic.size(); ++i) {
            if (bestNetworkData.bic[i] == std::numeric_limits<double>::infinity()) {
                continue;
            }
            std::cout << i << ", " << bestNetworkData.logl[i] << ", " << bestNetworkData.bic[i] << ", " << bestNetworkData.newick[i] << "\n";
            
            std::ofstream outfile(ann_network.options.output_file + "_" + std::to_string(i) + "_reticulations.nw");
            outfile << bestNetworkData.newick[i] << "\n";
            outfile.close();
        }
        std::ofstream outfile(ann_network.options.output_file);
        outfile << bestNetworkData.newick[bestNetworkData.best_n_reticulations] << "\n";
        outfile.close();
    }
}

void run_random(const NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng) {
    std::uniform_int_distribution<long> dist(0, RAND_MAX);
    BestNetworkData bestNetworkData(netraxOptions.max_reticulations);

    #ifdef USE_OPENMP
    netrax::AnnotatedNetwork ann_network_proto = build_random_annotated_network(netraxOptions, 42);
    std::vector<AnnotatedNetwork> ann_network_thread(omp_get_max_threads(), AnnotatedNetwork(ann_network_proto));
    #else
    std::vector<AnnotatedNetwork> ann_network_thread;
    #endif

    Statistics totalStats;
    std::vector<MoveType> allTypes = {MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::HeadMove, MoveType::TailMove, MoveType::RSPRMove, MoveType::DeltaPlusMove, MoveType::ArcInsertionMove, MoveType::DeltaMinusMove, MoveType::ArcRemovalMove};
    for (MoveType type : allTypes) {
        totalStats.moves_taken[type] = 0;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t start_reticulations = 0;
    size_t n_iterations = 0;
    // random start networks
    if (netraxOptions.num_random_start_networks > 0) {
        while (true) {
            n_iterations++;
            int seed = dist(rng);
            if (can_write()) {
                std::cout << "Starting with new random network " << n_iterations << " with " << start_reticulations << " reticulations, tree seed = " << seed << ".\n";
            }
            netrax::AnnotatedNetwork ann_network = build_random_annotated_network(netraxOptions, instance, seed);
            init_annotated_network(ann_network, rng);
            add_extra_reticulations(ann_network, start_reticulations);

            wavesearch(ann_network, &bestNetworkData, typesBySpeed, ann_network_thread);
            if (can_write()) {
                std::cout << " Inferred " << ann_network.network.num_reticulations() << " reticulations, logl = " << computeLoglikelihood(ann_network) << ", bic = " << scoreNetwork(ann_network) << "\n";
            }
            for (MoveType type : allTypes) {
                totalStats.moves_taken[type] += ann_network.stats.moves_taken[type];
            }
            //std::cout << "Ending with new random tree with " << ann_network.network.num_reticulations() << " reticulations.\n";
            if (netraxOptions.timeout > 0) {
                auto act_time = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(act_time - start_time).count() >= netraxOptions.timeout) {
                    break;
                }
            } else if (n_iterations >= netraxOptions.num_random_start_networks) {
                break;
            }
        }
    }

    // TODO: Get rid of the code duplication here
    // parsimony start networks
    n_iterations = 0;
    if (netraxOptions.num_parsimony_start_networks > 0) {
        while (true) {
            n_iterations++;
            int seed = dist(rng);
            if (can_write()) {
                std::cout << "Starting with new parsimony tree " << n_iterations << " with " << start_reticulations << " reticulations, tree seed = " << seed << ".\n";
            }
            netrax::AnnotatedNetwork ann_network = build_parsimony_annotated_network(netraxOptions, instance, seed);
            init_annotated_network(ann_network, rng);
            add_extra_reticulations(ann_network, start_reticulations);
            wavesearch(ann_network, &bestNetworkData, typesBySpeed, ann_network_thread);
            if (can_write()) {
                std::cout << " Inferred " << ann_network.network.num_reticulations() << " reticulations, logl = " << computeLoglikelihood(ann_network) << ", bic = " << scoreNetwork(ann_network) << "\n";
            }
            for (MoveType type : allTypes) {
                totalStats.moves_taken[type] += ann_network.stats.moves_taken[type];
            }
            //std::cout << "Ending with new parsimony tree with " << ann_network.network.num_reticulations() << " reticulations.\n";
            if (netraxOptions.timeout > 0) {
                auto act_time = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(act_time - start_time).count() >= netraxOptions.timeout) {
                    break;
                }
            } else if (n_iterations >= netraxOptions.num_parsimony_start_networks) {
                break;
            }
        }
    }

    if (can_write()) {
        std::cout << "\nAggregated statistics on which moves were taken:\n";
        for (const MoveType& type : typesBySpeed) {
            std::cout << toString(type) << ": " << totalStats.moves_taken[type] << "\n";
        }
        std::cout << "\n";

        std::cout << "Best inferred network has " << bestNetworkData.best_n_reticulations << " reticulations, logl = " << bestNetworkData.logl[bestNetworkData.best_n_reticulations] << ", bic = " << bestNetworkData.bic[bestNetworkData.best_n_reticulations] << "\n";
        std::cout << "Best inferred network is: \n";
        std::cout << bestNetworkData.newick[bestNetworkData.best_n_reticulations] << "\n";

        std::cout << "n_reticulations, logl, bic, newick\n";
        for (size_t i = 0; i < bestNetworkData.bic.size(); ++i) {
            if (bestNetworkData.bic[i] == std::numeric_limits<double>::infinity()) {
                continue;
            }
            std::cout << i << ", " << bestNetworkData.logl[i] << ", " << bestNetworkData.bic[i] << ", " << bestNetworkData.newick[i] << "\n";
            
            std::ofstream outfile(netraxOptions.output_file + "_" + std::to_string(i) + "_reticulations.nw");
            outfile << bestNetworkData.newick[i] << "\n";
            outfile.close();
        }
        std::ofstream outfile(netraxOptions.output_file);
        outfile << bestNetworkData.newick[bestNetworkData.best_n_reticulations] << "\n";
        outfile.close();
    }
}

}
