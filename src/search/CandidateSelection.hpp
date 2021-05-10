#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../optimization/NetworkState.hpp"
#include "ScoreImprovement.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/PseudoLoglikelihood.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../moves/MoveType.hpp"
#include "../moves/Moves.hpp"
#include "../io/NetworkIO.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/Optimization.hpp"

namespace netrax {

inline bool needsRecompute(AnnotatedNetwork& ann_network, const ArcRemovalMove& move) {
    return (ann_network.network.reticulation_nodes[ann_network.network.num_reticulations() - 1]->clv_index != move.v_clv_index);
}
inline bool needsRecompute(AnnotatedNetwork& ann_network, const ArcInsertionMove& move) {
    return false;
}
inline bool needsRecompute(AnnotatedNetwork& ann_network, const RSPRMove& move) {
    return false;
}
inline bool needsRecompute(AnnotatedNetwork& ann_network, const RNNIMove& move) {
    return false;
}

template <typename T>
struct ScoreItem {
    T move;
    double bicScore;
};

template <typename T>
void prefilterCandidates(AnnotatedNetwork& ann_network, std::vector<T>& candidates, bool silent = true, bool print_progress = true) {
    if (candidates.empty()) {
        return;
    }

    if (ParallelContext::local_proc_id() == 0) {
        if (print_progress) std::cout << "MoveType: " << toString(candidates[0].moveType) << " (" << candidates.size() << ")" << ", we currently have " << ann_network.network.num_reticulations() << " reticulations and BIC " << scoreNetwork(ann_network) << "\n";
    }

    float progress = 0.0;
    int barWidth = 70;

    double brlen_smooth_factor = 0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int max_iters_outside = max_iters;
    int radius = 1;
    double old_bic = scoreNetwork(ann_network);

    double best_bic = std::numeric_limits<double>::infinity();

    assert(computeLoglikelihood(ann_network) == computeLoglikelihood(ann_network, 0, 1));

    NetworkState oldState = extract_network_state(ann_network);

    assert(computeLoglikelihood(ann_network) == computeLoglikelihood(ann_network, 0, 1));

    std::vector<ScoreItem<T> > scores(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {        
        // progress bar code taken from https://stackoverflow.com/a/14539953/14557921
        if (print_progress && ParallelContext::local_proc_id() == 0) {
            progress = (float) (i+1) / candidates.size();
            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
        }

        assert(computeLoglikelihood(ann_network) == computeLoglikelihood(ann_network, 0, 1));

        assert(network_states_equal(extract_network_state(ann_network), oldState));
        T move(candidates[i]);
        bool recompute_from_scratch = needsRecompute(ann_network, move);

        assert(checkSanity(ann_network, move));

        performMove(ann_network, move);
        if (recompute_from_scratch) { // TODO: This is a hotfix that just masks some bugs. Fix the bugs properly.
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        assert(computeLoglikelihood(ann_network) == computeLoglikelihood(ann_network, 0, 1));
        optimizeReticulationProbs(ann_network);

        assert(computeLoglikelihood(ann_network) == computeLoglikelihood(ann_network, 0, 1));

        if (!hasBadReticulation(ann_network) || ((move.moveType != MoveType::DeltaPlusMove) && (move.moveType != MoveType::ArcInsertionMove))) {
            std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
            assert(!brlen_opt_candidates.empty());
            
            //std::cout << "thread " << ParallelContext::local_proc_id() << ", " << "before brlen opt, candidate no. " << i << "\n";
            //LikelihoodVariant old_variant = ann_network.options.likelihood_variant;
            //ann_network.options.likelihood_variant = LikelihoodVariant::SARAH_PSEUDO;
            optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates, true);
            //ann_network.options.likelihood_variant = old_variant;
            optimizeReticulationProbs(ann_network);
            //std::cout << "thread " << ParallelContext::local_proc_id() << ", " << "after brlen opt, candidate no. " << i << "\n";
            /*
            if (move->moveType == MoveType::ArcInsertionMove || move->moveType == MoveType::DeltaPlusMove) {
                optimize_branches(ann_network, max_iters, 1, radius, brlen_opt_candidates, false);
            } else {
                optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates, true);
            }*/
        }

        double bicScore = scoreNetwork(ann_network);

        scores[i] = ScoreItem<T>{candidates[i], bicScore};

        for (size_t j = 0; j < ann_network.network.num_nodes(); ++j) {
            assert(ann_network.network.nodes_by_index[j]->clv_index == j);
        }

        if (bicScore < best_bic) {
            best_bic = bicScore;
            ann_network.last_accepted_move_edge_orig_idx = move.edge_orig_idx;
        }

        if (old_bic/bicScore > ann_network.options.greedy_factor) {
            candidates[0] = candidates[i];
            candidates.resize(1);
            apply_network_state(ann_network, oldState);
            if (print_progress && ParallelContext::local_proc_id() == 0) {
                std::cout << std::endl;
            }
            return;
        }

        apply_network_state(ann_network, oldState);
    }

    if (print_progress && ParallelContext::local_proc_id() == 0) { 
        std::cout << std::endl;
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    double cutoff_bic = best_bic;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (ParallelContext::local_proc_id() == 0) {
            if (!silent) std::cout << "prefiltered candidate " << i + 1 << "/" << candidates.size() << " has BIC: " << scores[i].bicScore << "\n";
        }
        if (scores[i].bicScore <= cutoff_bic) {
            candidates[newSize] = scores[i].move;
            newSize++;
        }
    }
    if (ParallelContext::local_proc_id() == 0) {
        if (!silent) std::cout << "New size after prefiltering: " << newSize << " vs. " << candidates.size() << "\n";
    }

    candidates.resize(newSize);

    for (size_t i = 0; i < candidates.size(); ++i) {
        assert(checkSanity(ann_network, candidates[i]));
    }
}

template <typename T>
bool rankCandidates(AnnotatedNetwork& ann_network, std::vector<T> candidates, NetworkState* state, bool enforce, bool silent = true) {
    if (candidates.empty()) {
        return false;
    }
    if (!ann_network.options.no_prefiltering) {
        prefilterCandidates(ann_network, candidates, true);
    }

    if (ParallelContext::local_proc_id() == 0) {
        if (!silent) std::cout << "MoveType: " << toString(candidates[0].moveType) << "\n";
    }

    double brlen_smooth_factor = 1.0;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int max_iters_outside = max_iters;
    int radius = 1;

    double old_bic = scoreNetwork(ann_network);
    double best_bic = old_bic;

    if (enforce) {
        best_bic = std::numeric_limits<double>::infinity();
    }

    bool found_better = false;

    NetworkState oldState = extract_network_state(ann_network);

    std::vector<ScoreItem<T> > scores(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        T move(candidates[i]);
        bool recompute_from_scratch = needsRecompute(ann_network, move);

        assert(checkSanity(ann_network, move));

        performMove(ann_network, move);
        if (recompute_from_scratch) { // TODO: This is a hotfix that just masks some bugs. Fix the bugs properly.
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
        optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates);
        optimizeReticulationProbs(ann_network);

        double bicScore = scoreNetwork(ann_network);

        if (bicScore < best_bic) {
            best_bic = bicScore;
            ann_network.last_accepted_move_edge_orig_idx = move.edge_orig_idx;
            if (found_better) {
                extract_network_state(ann_network, *state);
            } else {
                *state = extract_network_state(ann_network);
            }
            found_better = true;
        }


        if (old_bic/bicScore > ann_network.options.greedy_factor) {
            candidates[0] = candidates[i];
            candidates.resize(1);
            apply_network_state(ann_network, oldState);
            return found_better;
        }

        scores[i] = ScoreItem<T>{candidates[i], bicScore};

        apply_network_state(ann_network, oldState);
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (ParallelContext::local_proc_id() == 0) {
            if (!silent) std::cout << "candidate " << i + 1 << "/" << candidates.size() << " has BIC: " << scores[i].bicScore << "\n";
        }
        if (scores[i].bicScore < old_bic) {
            candidates[newSize] = scores[i].move;
            newSize++;
        }
    }

    candidates.resize(newSize);

    return found_better;
}

template <typename T>
double applyBestCandidate(AnnotatedNetwork& ann_network, std::vector<T> candidates, double* best_score, BestNetworkData* bestNetworkData, bool enforce, bool silent) {
    NetworkState state = extract_network_state(ann_network);
    bool found_better_state = rankCandidates(ann_network, candidates, &state, enforce, true);
    double old_score = scoreNetwork(ann_network);

    if (found_better_state) {
        apply_network_state(ann_network, state);

        optimizeAllNonTopology(ann_network);
        double logl = computeLoglikelihood(ann_network);
        double bic_score = bic(ann_network, logl);
        double aic_score = aic(ann_network, logl);
        double aicc_score = aicc(ann_network, logl);

        double pseudo;
        if (ann_network.options.computePseudo) {
            pseudo = computePseudoLoglikelihood(ann_network);
        }
        if (ParallelContext::local_proc_id() == 0) {
            if (ann_network.options.computePseudo) {
                std::cout << "pseudo-loglh: " << pseudo << "\n";
                std::cout << "pseudo-bic: " << bic(ann_network, pseudo) << "\n";
            }

            if (!silent) std::cout << " Took " << toString(candidates[0].moveType) << "\n";
            if (!silent) std::cout << "  Logl: " << logl << ", BIC: " << bic_score << ", AIC: " << aic_score << ", AICc: " << aicc_score <<  "\n";
            if (!silent) std::cout << "  param_count: " << get_param_count(ann_network) << ", sample_size:" << get_sample_size(ann_network) << "\n";
            if (!silent) std::cout << "  num_reticulations: " << ann_network.network.num_reticulations() << "\n";
            if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
        }
        ann_network.stats.moves_taken[candidates[0].moveType]++;

        check_score_improvement(ann_network, best_score, bestNetworkData);

        if (!enforce) {
            if (scoreNetwork(ann_network) > old_score) {
                throw std::runtime_error("Something went wrong in the network search. Suddenly, BIC is worse!");
            }
        }
    }

    return computeLoglikelihood(ann_network);
}

}