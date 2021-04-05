#include "NetworkSearch.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "../likelihood/mpreal.h"
#include "../graph/AnnotatedNetwork.hpp"
#include "../io/NetworkIO.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../optimization/TopologyOptimization.hpp"
#include "../optimization/ModelOptimization.hpp"
#include "../optimization/ReticulationOptimization.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/Moves.hpp"
#include "../optimization/MoveType.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../optimization/NetworkState.hpp"

namespace netrax {

struct ScoreImprovementResult {
    bool local_improved = false;
    bool global_improved = false;
};

bool logl_stays_same(AnnotatedNetwork& ann_network) {
    std::cout << "displayed trees before:\n";
    for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
        size_t n_trees = ann_network.pernode_displayed_tree_data[i][ann_network.network.root->clv_index].num_active_displayed_trees;
        for (size_t j = 0; j < n_trees; ++j) {
            DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[i][ann_network.network.root->clv_index].displayed_trees[j];
            std::cout << "logl: " << tree.treeLoglData.tree_logl << ", logprob: " << tree.treeLoglData.tree_logprob << "\n";
        }
    }
    double incremental = netrax::computeLoglikelihood(ann_network, 1, 1);
    std::cout << "displayed trees in between:\n";
    for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
        size_t n_trees = ann_network.pernode_displayed_tree_data[i][ann_network.network.root->clv_index].num_active_displayed_trees;
        for (size_t j = 0; j < n_trees; ++j) {
            DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[i][ann_network.network.root->clv_index].displayed_trees[j];
            std::cout << "logl: " << tree.treeLoglData.tree_logl << ", logprob: " << tree.treeLoglData.tree_logprob << "\n";
        }
    }
    double normal = netrax::computeLoglikelihood(ann_network, 0, 1);
    std::cout << "displayed trees after:\n";
    for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
        size_t n_trees = ann_network.pernode_displayed_tree_data[i][ann_network.network.root->clv_index].num_active_displayed_trees;
        for (size_t j = 0; j < n_trees; ++j) {
            DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[i][ann_network.network.root->clv_index].displayed_trees[j];
            std::cout << "logl: " << tree.treeLoglData.tree_logl << ", logprob: " << tree.treeLoglData.tree_logprob << "\n";
        }
    }

    std::cout << "incremental: " << incremental << "\n";
    std::cout << "normal: " << normal << "\n";
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

        if (score_after < score_before - ann_network.options.score_epsilon && extremeOpt) {
            gotBetter = true;
        }
    }

    assert(logl_stays_same(ann_network));
}

void printDisplayedTrees(AnnotatedNetwork& ann_network) {
    std::vector<std::pair<std::string, double>> displayed_trees;
    if (ann_network.network.num_reticulations() == 0) {
        std::string newick = netrax::toExtendedNewick(ann_network);
        displayed_trees.emplace_back(std::make_pair(newick, 1.0));
    } else {
        size_t n_trees = ann_network.pernode_displayed_tree_data[0][ann_network.network.root->clv_index].num_active_displayed_trees;
        for (size_t j = 0; j < n_trees; ++j) {
            DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[0][ann_network.network.root->clv_index].displayed_trees[j];
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

ScoreImprovementResult check_score_improvement(AnnotatedNetwork& ann_network, double* local_best, BestNetworkData* bestNetworkData) {
    bool local_improved = false;
    bool global_improved = false;
    
    double new_score = scoreNetwork(ann_network);
    double score_diff = bestNetworkData->bic[ann_network.network.num_reticulations()] - new_score;
    if (score_diff > 0.0001) {
        bestNetworkData->bic[ann_network.network.num_reticulations()] = new_score;
        bestNetworkData->logl[ann_network.network.num_reticulations()] = computeLoglikelihood(ann_network, 1, 1);
        bestNetworkData->newick[ann_network.network.num_reticulations()] = toExtendedNewick(ann_network);
        
        local_improved = true;
        if (new_score < *local_best) {
            std::cout << "SCORE DIFF: " << score_diff << "\n";
            std::cout << "OLD LOCAL BEST SCORE WAS: " << *local_best << "\n";
            *local_best = new_score;
            std::cout << "IMPROVED LOCAL BEST SCORE FOUND SO FAR: " << new_score << "\n\n";

            double old_global_best = bestNetworkData->bic[bestNetworkData->best_n_reticulations];

            if (new_score < old_global_best) {
                if (hasBadReticulation(ann_network)) {
                    std::cout << "Network contains BAD RETICULATIONS. Not updating the global best found network and score.\n";
                } else {
                    bestNetworkData->best_n_reticulations = ann_network.network.num_reticulations();
                    global_improved = true;
                    std::cout << "OLD GLOBAL BEST SCORE WAS: " << old_global_best << "\n";
                    std::cout << "IMPROVED GLOBAL BEST SCORE FOUND SO FAR: " << new_score << "\n\n";
                    writeNetwork(ann_network, ann_network.options.output_file);
                    std::cout << toExtendedNewick(ann_network) << "\n";
                    std::cout << "Better network written to " << ann_network.options.output_file << "\n";
                    printDisplayedTrees(ann_network);
                }
            } else {
                std::cout << "REMAINED GLOBAL BEST SCORE FOUND SO FAR: " << old_global_best << "\n\n";
            }
        } else {
            std::cout << "REMAINED LOCAL BEST SCORE FOUND SO FAR: " << *local_best << "\n\n";
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

bool isComplexityChangingMove(MoveType& moveType) {
    return (moveType == MoveType::ArcRemovalMove || moveType == MoveType::ArcInsertionMove || moveType == MoveType::DeltaMinusMove || moveType == MoveType::DeltaPlusMove);
}

template <typename T>
void prefilterCandidates(AnnotatedNetwork& ann_network, std::vector<T>& candidates, bool silent = false) {
    if (candidates.empty()) {
        return;
    }
    double brlen_smooth_factor = 0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = 1;
    double old_bic = scoreNetwork(ann_network);

    NetworkState oldState = extract_network_state(ann_network);

    std::vector<ScoreItem<T> > scores(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        T move = candidates[i];
        assert(checkSanity(ann_network, move));
        performMove(ann_network, move);
        optimizeReticulationProbs(ann_network);
        
        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
        double worstScore = getWorstReticulationScore(ann_network);
        double bicScore = scoreNetwork(ann_network);

        scores[i] = ScoreItem<T>{move, worstScore, bicScore};

        if (isComplexityChangingMove(move.moveType)) {
            apply_network_state(ann_network, oldState, true);
        } else {
            undoMove(ann_network, move);
            apply_network_state(ann_network, oldState, false);
        }

        assert(checkSanity(ann_network, move));

        if (bicScore < old_bic) {
            candidates[0] = candidates[i];
            candidates.resize(1);
            return;
        }
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        if (lhs.bicScore == rhs.bicScore) {
            return lhs.worstScore > rhs.worstScore;
        }
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    double cutoff_bic = scores[scores.size()*0.1].bicScore;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!silent) std::cout << "prefiltered candidate " << i + 1 << "/" << candidates.size() << " has worst score " << scores[i].worstScore << ", BIC: " << scores[i].bicScore << "\n";
        if (scores[i].bicScore <= cutoff_bic) {
            candidates[newSize] = scores[i].move;
            newSize++;
        }
    }
    if (!silent) std::cout << "New size after prefiltering: " << newSize << " vs. " << candidates.size() << "\n";

    candidates.resize(newSize);
}

template <typename T>
void rankCandidates(AnnotatedNetwork& ann_network, std::vector<T>& candidates, bool silent = false) {
    if (candidates.empty()) {
        return;
    }

    if (candidates[0].moveType != MoveType::ArcRemovalMove) {
        prefilterCandidates(ann_network, candidates, true);
    }

    std::cout << "MoveType: " << toString(candidates[0].moveType) << "\n";

    double brlen_smooth_factor = 0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = 1;

    double old_bic = scoreNetwork(ann_network);

    NetworkState oldState = extract_network_state(ann_network);

    std::vector<ScoreItem<T> > scores(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        T move = candidates[i];
        performMove(ann_network, move);

        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
        optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
        optimizeReticulationProbs(ann_network);
        
        double worstScore = getWorstReticulationScore(ann_network);
        double bicScore = scoreNetwork(ann_network);

        scores[i] = ScoreItem<T>{move, worstScore, bicScore};

        if (isComplexityChangingMove(move.moveType)) {
            apply_network_state(ann_network, oldState, true);
        } else {
            undoMove(ann_network, move);
            apply_network_state(ann_network, oldState, true);
        }

        if (bicScore < old_bic) {
            candidates[0] = candidates[i];
            candidates.resize(1);
            return;
        }
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        if (lhs.bicScore == rhs.bicScore) {
            return lhs.worstScore > rhs.worstScore;
        }
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!silent) std::cout << "candidate " << i + 1 << "/" << candidates.size() << " has worst score " << scores[i].worstScore << ", BIC: " << scores[i].bicScore << "\n";
        if (scores[i].bicScore < old_bic) {
            candidates[newSize] = scores[i].move;
            newSize++;
        }
    }

    candidates.resize(newSize);
}

template <typename T>
double applyBestCandidate(AnnotatedNetwork& ann_network, std::vector<T> candidates, bool silent = false) {
    double brlen_smooth_factor = 0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = 1;

    rankCandidates(ann_network, candidates, true);

    if (!candidates.empty()) {
        T move = candidates[0];
        performMove(ann_network, move);

        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
        optimize_branches(ann_network, max_iters, radius, brlen_opt_candidates);
        optimizeReticulationProbs(ann_network);

        double logl = computeLoglikelihood(ann_network);
        double bic_score = bic(ann_network, logl);
        double aic_score = aic(ann_network, logl);
        double aicc_score = aicc(ann_network, logl);

        if (!silent) std::cout << " Took " << toString(move.moveType) << "\n";
        if (!silent) std::cout << "  Logl: " << logl << ", BIC: " << bic_score << ", AIC: " << aic_score << ", AICc: " << aicc_score <<  "\n";
        if (!silent) std::cout << "  param_count: " << get_param_count(ann_network) << ", sample_size:" << get_sample_size(ann_network) << "\n";
        if (!silent) std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
        if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
        ann_network.stats.moves_taken[move.moveType]++;
    }

    return computeLoglikelihood(ann_network);
}

double forceApplyArcInsertion(AnnotatedNetwork& ann_network) {
    std::vector<ArcInsertionMove> candidates = possibleArcInsertionMoves(ann_network);
    return applyBestCandidate(ann_network, candidates);
}

double optimizeEverythingRun(AnnotatedNetwork & ann_network, std::vector<MoveType>& typesBySpeed, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, const std::chrono::high_resolution_clock::time_point& start_time, bool greedy = true) {
    unsigned int type_idx = 0;
    unsigned int max_seconds = ann_network.options.timeout;
    double best_score = scoreNetwork(ann_network);
    do {
        while (ann_network.network.num_reticulations() == 0
            && (typesBySpeed[type_idx] == MoveType::DeltaMinusMove || typesBySpeed[type_idx] == MoveType::ArcRemovalMove)) {
            type_idx++;
            if (type_idx >= typesBySpeed.size()) {
                break;
            }
        }
        if (type_idx >= typesBySpeed.size()) {
            break;
        }
        while (ann_network.network.num_reticulations() == ann_network.options.max_reticulations
            && (typesBySpeed[type_idx] == MoveType::DeltaPlusMove || typesBySpeed[type_idx] == MoveType::ArcInsertionMove)) {
            type_idx++;
            if (type_idx >= typesBySpeed.size()) {
                break;
            }
        }
        if (type_idx >= typesBySpeed.size()) {
            break;
        }
        double old_score = scoreNetwork(ann_network);
        //optimizeTopology(ann_network, typesBySpeed[type_idx], start_state_to_reuse, best_state_to_reuse, greedy, false, false, 1);

        switch (typesBySpeed[type_idx]) {
        case MoveType::RNNIMove:
            applyBestCandidate(ann_network, possibleRNNIMoves(ann_network));
            break;
        case MoveType::RSPRMove:
            applyBestCandidate(ann_network, possibleRSPRMoves(ann_network));
            break;
        case MoveType::RSPR1Move:
            applyBestCandidate(ann_network, possibleRSPR1Moves(ann_network));
            break;
        case MoveType::HeadMove:
            applyBestCandidate(ann_network, possibleHeadMoves(ann_network));
            break;
        case MoveType::TailMove:
            applyBestCandidate(ann_network, possibleTailMoves(ann_network));
            break;
        case MoveType::ArcInsertionMove:
            applyBestCandidate(ann_network, possibleArcInsertionMoves(ann_network));
            break;
        case MoveType::DeltaPlusMove:
            applyBestCandidate(ann_network, possibleDeltaPlusMoves(ann_network));
            break;
        case MoveType::ArcRemovalMove:
            applyBestCandidate(ann_network, possibleArcRemovalMoves(ann_network));
            break;
        case MoveType::DeltaMinusMove:
            //std::cout << exportDebugInfo(ann_network) << "\n";
            applyBestCandidate(ann_network, possibleDeltaMinusMoves(ann_network));
            break;
        default:
            throw std::runtime_error("Invalid move type");
        }

        double new_score = scoreNetwork(ann_network);
        if (new_score < old_score) { // score got better
            new_score = scoreNetwork(ann_network);
            best_score = new_score;

            type_idx = 0; // go back to fastest move type        
        } else { // try next-slower move type
            type_idx++;
        }
        assert(new_score <= old_score + ann_network.options.score_epsilon);

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

void wavesearch(AnnotatedNetwork& ann_network, BestNetworkData* bestNetworkData, std::mt19937& rng) {
    NetworkState start_state_to_reuse = extract_network_state(ann_network, false);
    NetworkState best_state_to_reuse = extract_network_state(ann_network, false);
    auto start_time = std::chrono::high_resolution_clock::now();
    double best_score = std::numeric_limits<double>::infinity();
    ScoreImprovementResult score_improvement;

    //std::vector<MoveType> typesBySpeed = {MoveType::ArcRemovalMove, MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove, MoveType::ArcInsertionMove};
    //std::vector<MoveType> typesBySpeed = {MoveType::ArcRemovalMove, MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove, MoveType::DeltaPlusMove};

    std::vector<MoveType> typesBySpeed = {MoveType::DeltaMinusMove, MoveType::RNNIMove, MoveType::DeltaPlusMove};

    std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";

    optimizeAllNonTopology(ann_network, true);
    std::cout << "Initial network after modelopt+brlenopt+reticulation opt is:\n" << toExtendedNewick(ann_network) << "\n\n";
    std::string best_network = toExtendedNewick(ann_network);
    score_improvement = check_score_improvement(ann_network, &best_score, bestNetworkData);

    optimizeEverythingRun(ann_network, typesBySpeed, start_state_to_reuse, best_state_to_reuse, start_time, true);
    score_improvement = check_score_improvement(ann_network, &best_score, bestNetworkData);
}

void run_single_start_waves(NetraxOptions& netraxOptions, std::mt19937& rng) {
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions);
    init_annotated_network(ann_network, rng);
    BestNetworkData bestNetworkData(ann_network.options.max_reticulations);
    wavesearch(ann_network, &bestNetworkData, rng);

    std::cout << "Statistics on which moves were taken:\n";
    for (const auto& entry : ann_network.stats.moves_taken) {
        std::cout << toString(entry.first) << ": " << entry.second << "\n";
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

void run_random(NetraxOptions& netraxOptions, std::mt19937& rng) {
    BestNetworkData bestNetworkData(netraxOptions.max_reticulations);

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t start_reticulations = 0;
    size_t n_iterations = 0;
    // random start networks
    while (true) {
        n_iterations++;
        std::cout << "Starting with new random network with " << start_reticulations << " reticulations.\n";
        netrax::AnnotatedNetwork ann_network = build_random_annotated_network(netraxOptions);
        init_annotated_network(ann_network, rng);
        add_extra_reticulations(ann_network, start_reticulations);

        wavesearch(ann_network, &bestNetworkData, rng);
        if (netraxOptions.timeout > 0) {
            auto act_time = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(act_time - start_time).count() >= netraxOptions.timeout) {
                break;
            }
        } else if (n_iterations >= netraxOptions.num_random_start_networks) {
            break;
        }
    }

    // TODO: Get rid of the code duplication here
    // parsimony start networks
    n_iterations = 0;
    while (true) {
        n_iterations++;
        std::cout << "Starting with new parsimony tree with " << start_reticulations << " reticulations.\n";
        netrax::AnnotatedNetwork ann_network = build_parsimony_annotated_network(netraxOptions);
        init_annotated_network(ann_network, rng);
        add_extra_reticulations(ann_network, start_reticulations);
        wavesearch(ann_network, &bestNetworkData, rng);
        if (netraxOptions.timeout > 0) {
            auto act_time = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(act_time - start_time).count() >= netraxOptions.timeout) {
                break;
            }
        } else if (n_iterations >= netraxOptions.num_parsimony_start_networks) {
            break;
        }
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
        
        std::ofstream outfile(netraxOptions.output_file + "_" + std::to_string(i) + "_reticulations.nw");
        outfile << bestNetworkData.newick[i] << "\n";
        outfile.close();
    }
    std::ofstream outfile(netraxOptions.output_file);
    outfile << bestNetworkData.newick[bestNetworkData.best_n_reticulations] << "\n";
    outfile.close();
}

}
