#include "CandidateSelection.hpp"
#include "../helper/Helper.hpp"

#include "../optimization/NetworkState.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/PseudoLoglikelihood.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../moves/Moves.hpp"
#include "../io/NetworkIO.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/Optimization.hpp"
#include "../colormod.h" // namespace Color

namespace netrax {

template <typename T>
struct ScoreItem {
    T item;
    double bicScore;
};

double trim(double x, int digitsAfterComma) {
    double factor = pow(10, digitsAfterComma);
    return (double)((int)(x*factor))/factor;
}

void switchLikelihoodVariant(AnnotatedNetwork& ann_network, LikelihoodVariant newVariant) {
    return;
    if (ann_network.options.likelihood_variant == newVariant) {
        return;
    }
    ann_network.options.likelihood_variant = newVariant;
    ann_network.cached_logl_valid = false;
}

template <typename T>
double prefilterCandidates(AnnotatedNetwork& ann_network, std::vector<T>& candidates, bool silent = true, bool print_progress = true, bool need_best_bic = false) {
    std::unordered_set<size_t> promisingNodes;
    
    if (candidates.empty()) {
        return scoreNetwork(ann_network);
    }

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (print_progress) std::cout << "MoveType: " << toString(candidates[0].moveType) << " (" << candidates.size() << ")" << ", we currently have " << ann_network.network.num_reticulations() << " reticulations and BIC " << scoreNetwork(ann_network) << "\n";
    }

    std::vector<double> nodeScore(ann_network.network.num_nodes(), std::numeric_limits<double>::infinity());

    LikelihoodVariant old_variant = ann_network.options.likelihood_variant;
    switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);

    float progress = 0.0;
    int barWidth = 70;

    double old_bic = scoreNetwork(ann_network);

    switchLikelihoodVariant(ann_network, old_variant);
    double real_old_bic = scoreNetwork(ann_network);
    switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);

    double best_bic = old_bic; //std::numeric_limits<double>::infinity();
    double best_real_bic = real_old_bic;

    NetworkState oldState = extract_network_state(ann_network);

    for (size_t i = 0; i < candidates.size(); ++i) {        
        // progress bar code taken from https://stackoverflow.com/a/14539953/14557921
        if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) {
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

        T move(candidates[i]);
        bool recompute_from_scratch = needsRecompute(ann_network, move);

        assert(checkSanity(ann_network, move));

        performMove(ann_network, move);
        if (recompute_from_scratch && ann_network.options.likelihood_variant != LikelihoodVariant::SARAH_PSEUDO) { // TODO: This is a hotfix that just masks some bugs. Fix the bugs properly.
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        if (move.moveType == MoveType::ArcInsertionMove || move.moveType == MoveType::DeltaPlusMove) {
            optimizeReticulationProbs(ann_network);
            std::unordered_set<size_t> brlenopt_candidates;
            brlenopt_candidates.emplace(((ArcInsertionMove*) &move)->wanted_uv_pmatrix_index);
            optimizeBranchesCandidates(ann_network, brlenopt_candidates);
        }

        double bicScore = scoreNetwork(ann_network);
        nodeScore[move.node_orig_idx] = std::min(nodeScore[move.node_orig_idx], bicScore);

        for (size_t j = 0; j < ann_network.network.num_nodes(); ++j) {
            assert(ann_network.network.nodes_by_index[j]->clv_index == j);
        }

        if (bicScore < best_bic) {
            best_bic = bicScore;

            if (need_best_bic && ann_network.network.num_reticulations() > 0) {
                switchLikelihoodVariant(ann_network, old_variant);
                double actRealBIC = scoreNetwork(ann_network);
                if (actRealBIC < best_real_bic) {
                    best_real_bic = actRealBIC;
                    ann_network.last_accepted_move_edge_orig_idx = move.edge_orig_idx;
                    switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);
                }
            } else {
                best_real_bic = best_bic;
            }
        }

        double extra_offset = (ann_network.options.likelihood_variant == LikelihoodVariant::SARAH_PSEUDO) ? 0.01 : 0.0; // +0.01 because we tend to over-estimate with pseudologlikelihood
        if (old_bic/bicScore > ann_network.options.greedy_factor + extra_offset) {
            switchLikelihoodVariant(ann_network, old_variant);
            optimizeReticulationProbs(ann_network);
            double real_bicScore = scoreNetwork(ann_network);
            if (real_old_bic/real_bicScore > ann_network.options.greedy_factor) {
                candidates[0] = candidates[i];
                candidates.resize(1);
                undoMove(ann_network, move);
                apply_network_state(ann_network, oldState);
                if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) {
                    std::cout << std::endl;
                }
                switchLikelihoodVariant(ann_network, old_variant);
                return real_bicScore;
            } else {
                switchLikelihoodVariant(ann_network, LikelihoodVariant::SARAH_PSEUDO);
            }
        }

        undoMove(ann_network, move);
        if (move.moveType == MoveType::ArcRemovalMove || move.moveType == MoveType::DeltaMinusMove) {
            apply_network_state(ann_network, oldState);
        }
    }
    apply_network_state(ann_network, oldState);

    if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) { 
        std::cout << std::endl;
    }

    std::vector<ScoreItem<Node*> > scores(ann_network.network.num_nodes());
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        scores[i] = ScoreItem<Node*>{ann_network.network.nodes_by_index[i], nodeScore[i]};
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<Node*>& lhs, const ScoreItem<Node*>& rhs) {
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    size_t cutoff_pos = std::min(ann_network.options.prefilter_keep - 1, scores.size() - 1);

    double cutoff_bic = scores[cutoff_pos].bicScore; //best_bic;

    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        if (scores[i].bicScore == std::numeric_limits<double>::infinity()) {
            break;
        }
        if (scores[i].bicScore <= cutoff_bic) {
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                if (!silent) std::cout << "prefiltered node " << scores[i].item->clv_index << " has best BIC: " << scores[i].bicScore << "\n";
            }
            promisingNodes.emplace(scores[i].item->clv_index);
            newSize++;
        }

        if (newSize == ann_network.options.prefilter_keep) {
            break;
        }
    }

    size_t oldCandidatesSize = candidates.size();
    std::vector<T> newCandidates;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (promisingNodes.count(candidates[i].node_orig_idx) > 0) {
            newCandidates.emplace_back(candidates[i]);
        }
    }
    candidates = newCandidates;

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (print_progress) std::cout << "New size promising nodes after prefiltering: " << newSize << " vs. " << ann_network.network.num_nodes() << "\n";
        if (print_progress) std::cout << "New size candidates after prefiltering: " << candidates.size() << " vs. " << oldCandidatesSize << "\n";
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        assert(checkSanity(ann_network, candidates[i]));
    }
    switchLikelihoodVariant(ann_network, old_variant);

    return best_real_bic;
}


template <typename T>
void rankCandidates(AnnotatedNetwork& ann_network, std::vector<T>& candidates, bool silent = true, bool print_progress = true) {
    if (candidates.empty()) {
        return;
    }
    if (!ann_network.options.no_prefiltering) {
        prefilterCandidates(ann_network, candidates, true);
    }

    if (ParallelContext::master_rank() && ParallelContext::master_thread() && ann_network.options.no_prefiltering) {
        if (print_progress) std::cout << "MoveType: " << toString(candidates[0].moveType) << " (" << candidates.size() << ")" << ", we currently have " << ann_network.network.num_reticulations() << " reticulations and BIC " << scoreNetwork(ann_network) << "\n";
    }

    float progress = 0.0;
    int barWidth = 70;

    double brlen_smooth_factor = 1;//0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int max_iters_outside = 1;
    int radius = 1;
    double old_bic = scoreNetwork(ann_network);

    double best_bic = std::numeric_limits<double>::infinity();

    NetworkState oldState = extract_network_state(ann_network);

    std::vector<ScoreItem<T> > scores(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {        
        // progress bar code taken from https://stackoverflow.com/a/14539953/14557921
        if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) {
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

        T move(candidates[i]);
        bool recompute_from_scratch = needsRecompute(ann_network, move);

        assert(checkSanity(ann_network, move));

        performMove(ann_network, move);
        if (recompute_from_scratch && ann_network.options.likelihood_variant != LikelihoodVariant::SARAH_PSEUDO) { // TODO: This is a hotfix that just masks some bugs. Fix the bugs properly.
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        std::unordered_set<size_t> brlen_opt_candidates = brlenOptCandidates(ann_network, move);
        assert(!brlen_opt_candidates.empty());
        //add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
        optimize_branches(ann_network, max_iters, max_iters_outside, radius, brlen_opt_candidates);
        optimizeReticulationProbs(ann_network);

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
            undoMove(ann_network, move);
            apply_network_state(ann_network, oldState);
            if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << std::endl;
            }
            return;
        }
        undoMove(ann_network, move);
        if (move.moveType == MoveType::ArcRemovalMove || move.moveType == MoveType::DeltaMinusMove) {
            apply_network_state(ann_network, oldState);
        }
    }
    apply_network_state(ann_network, oldState);

    if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) { 
        std::cout << std::endl;
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    size_t cutoff_pos = std::min(ann_network.options.rank_keep - 1, scores.size() - 1);

    double cutoff_bic = scores[cutoff_pos].bicScore; //best_bic;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
            if (!silent) std::cout << "ranked candidate " << i + 1 << "/" << candidates.size() << " has BIC: " << scores[i].bicScore << "\n";
        }
        if (scores[i].bicScore <= cutoff_bic) {
            candidates[newSize] = scores[i].item;
            newSize++;
        }

        if (newSize == ann_network.options.rank_keep) {
            break;
        }
    }
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (print_progress) std::cout << "New size after ranking: " << newSize << " vs. " << candidates.size() << "\n";
    }

    candidates.resize(newSize);

    for (size_t i = 0; i < candidates.size(); ++i) {
        assert(checkSanity(ann_network, candidates[i]));
    }
}

template <typename T>
bool chooseCandidate(AnnotatedNetwork& ann_network, std::vector<T> candidates, NetworkState* state, bool enforce, bool silent = true, bool print_progress = true) {
    if (candidates.empty()) {
        return false;
    }
    rankCandidates(ann_network, candidates, silent, print_progress);

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (!silent) std::cout << "MoveType: " << toString(candidates[0].moveType) << "\n";
    }

    float progress = 0.0;
    int barWidth = 70;

    double brlen_smooth_factor = 1;//0.25;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int max_iters_outside = 1;//max_iters; // this says how often we will optimize the same branch again
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
        // progress bar code taken from https://stackoverflow.com/a/14539953/14557921
        if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) {
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
        T move(candidates[i]);
        bool recompute_from_scratch = needsRecompute(ann_network, move);

        assert(checkSanity(ann_network, move));

        performMove(ann_network, move);
        if (recompute_from_scratch && ann_network.options.likelihood_variant != LikelihoodVariant::SARAH_PSEUDO) { // TODO: This is a hotfix that just masks some bugs. Fix the bugs properly.
            computeLoglikelihood(ann_network, 0, 1); // this is needed because arc removal changes the reticulation indices
        }
        optimize_branches(ann_network, max_iters, max_iters_outside, radius);
        //optimizeReticulationProbs(ann_network);

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
            undoMove(ann_network, move);
            apply_network_state(ann_network, oldState);
            if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << std::endl;
            }
            return found_better;
        }

        scores[i] = ScoreItem<T>{candidates[i], bicScore};

        undoMove(ann_network, move);
        apply_network_state(ann_network, oldState);
    }
    if (print_progress && ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << std::endl;
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreItem<T>& lhs, const ScoreItem<T>& rhs) {
        return lhs.bicScore < rhs.bicScore;
    });

    size_t newSize = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
            if (!silent) std::cout << "candidate " << i + 1 << "/" << candidates.size() << " has BIC: " << scores[i].bicScore << "\n";
        }
        if (scores[i].bicScore < old_bic) {
            candidates[newSize] = scores[i].item;
            newSize++;
        }
    }

    candidates.resize(newSize);

    return found_better;
}

template <typename T>
double applyBestCandidate(AnnotatedNetwork& ann_network, std::vector<T> candidates, double* best_score, BestNetworkData* bestNetworkData, bool enforce, bool silent) {
    NetworkState state = extract_network_state(ann_network);
    bool found_better_state = chooseCandidate(ann_network, candidates, &state, enforce, silent);
    double old_score = scoreNetwork(ann_network);

    if (found_better_state) {
        apply_network_state(ann_network, state);

        double logl = computeLoglikelihood(ann_network);
        double bic_score = bic(ann_network, logl);
        double aic_score = aic(ann_network, logl);
        double aicc_score = aicc(ann_network, logl);

        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
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

    return scoreNetwork(ann_network);
}

double best_fast_improvement(AnnotatedNetwork& ann_network, MoveType type, const std::vector<MoveType>& typesBySpeed, double* best_score, BestNetworkData* bestNetworkData, bool silent, size_t min_radius, size_t max_radius) {
    bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(), MoveType::RSPR1Move) != typesBySpeed.end());
    bool delta_plus_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(), MoveType::DeltaPlusMove) != typesBySpeed.end());
    double score = scoreNetwork(ann_network);

    if (type == MoveType::RSPR1Move) {
        auto candidates2 = possibleRSPRMoves(ann_network, rspr1_present, min_radius, max_radius);
        score = prefilterCandidates(ann_network, candidates2, silent, true, true);
    } else if (type == MoveType::RSPRMove) {
        auto candidates2 = possibleRSPRMoves(ann_network, rspr1_present, min_radius, max_radius);
        score = prefilterCandidates(ann_network, candidates2, silent, true, true);
    } else if (type == MoveType::HeadMove) {
        auto candidates4 = possibleHeadMoves(ann_network, rspr1_present, min_radius, max_radius);
        score = prefilterCandidates(ann_network, candidates4, silent, true, true);
    } else if (type == MoveType::TailMove) {
        auto candidates5 = possibleTailMoves(ann_network, rspr1_present, min_radius, max_radius);
        score = prefilterCandidates(ann_network, candidates5, silent, true, true);    
    } else if (type == MoveType::ArcInsertionMove) {
        auto candidates6 = possibleArcInsertionMoves(ann_network, delta_plus_present, min_radius, max_radius);
        score = prefilterCandidates(ann_network, candidates6, silent, true, true);    
    } else if (type == MoveType::DeltaPlusMove) {
        auto candidates7 = possibleDeltaPlusMoves(ann_network, min_radius, max_radius);
        score = prefilterCandidates(ann_network, candidates7, silent, true, true);    
    }
    
    return score;
}

double applyBestCandidate(AnnotatedNetwork& ann_network, MoveType type, const std::vector<MoveType>& typesBySpeed, double* best_score, BestNetworkData* bestNetworkData, bool enforce, bool silent, size_t min_radius, size_t max_radius) {
    bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(), MoveType::RSPR1Move) != typesBySpeed.end());
    bool delta_plus_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(), MoveType::DeltaPlusMove) != typesBySpeed.end());
    switch (type) {
        case MoveType::RNNIMove:
            applyBestCandidate(ann_network, possibleRNNIMoves(ann_network), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::RSPRMove:
            applyBestCandidate(ann_network, possibleRSPRMoves(ann_network, rspr1_present, min_radius, max_radius), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::RSPR1Move:
            applyBestCandidate(ann_network, possibleRSPR1Moves(ann_network, min_radius, max_radius), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::HeadMove:
            applyBestCandidate(ann_network, possibleHeadMoves(ann_network, rspr1_present, min_radius, max_radius), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::TailMove:
            applyBestCandidate(ann_network, possibleTailMoves(ann_network, rspr1_present, min_radius, max_radius), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::ArcInsertionMove:
            applyBestCandidate(ann_network, possibleArcInsertionMoves(ann_network, delta_plus_present, min_radius, max_radius), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::DeltaPlusMove:
            applyBestCandidate(ann_network, possibleDeltaPlusMoves(ann_network, min_radius, max_radius), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::ArcRemovalMove:
            applyBestCandidate(ann_network, possibleArcRemovalMoves(ann_network), best_score, bestNetworkData, false, silent);
            break;
        case MoveType::DeltaMinusMove:
            applyBestCandidate(ann_network, possibleDeltaMinusMoves(ann_network), best_score, bestNetworkData, false, silent);
            break;
        default:
            throw std::runtime_error("Invalid move type");
    }
    return scoreNetwork(ann_network);
}

double fullSearch(AnnotatedNetwork& ann_network, MoveType type, const std::vector<MoveType>& typesBySpeed, double* best_score, BestNetworkData* bestNetworkData, bool silent) {
    double old_score = scoreNetwork(ann_network);
    
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        Color::Modifier blue(Color::FG_BLUE);
        Color::Modifier def(Color::FG_DEFAULT);
        std::cout << blue;
        std::cout << "\nStarting full search for move type: " << toString(type) << "\n";
        std::cout << def;
    }

    int step_size = 5;

    int best_max_distance = 0;
    if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove || type == MoveType::DeltaMinusMove) {
        best_max_distance = ann_network.options.max_rearrangement_distance;
    } else {
        // step 1: find best max distance
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
            std::cout << "step1: find best max distance\n";
        }
        int act_max_distance = 0;
        int old_max_distance = 0;
        double old_greedy_factor = ann_network.options.greedy_factor;
        ann_network.options.greedy_factor = 1.0;
        NetworkState oldState = extract_network_state(ann_network);
        while (act_max_distance < ann_network.options.max_rearrangement_distance) {
            act_max_distance = std::min(act_max_distance + step_size, ann_network.options.max_rearrangement_distance);
            double score = best_fast_improvement(ann_network, type, typesBySpeed, best_score, bestNetworkData, silent, old_max_distance, act_max_distance);
            if (score < old_score) {
                old_max_distance = act_max_distance + 1;
                best_max_distance = act_max_distance;
            } else {
                assert(score == old_score);
                break;
            }
            apply_network_state(ann_network, oldState);
        }
        ann_network.options.greedy_factor = old_greedy_factor;

        optimizeAllNonTopology(ann_network);
        check_score_improvement(ann_network, best_score, bestNetworkData);
    }

    // step 2: fast iterations mode, with the best max distance
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "\nstep 2: fast iterations mode, with the best max distance " << best_max_distance << "\n";
    }
    bool got_better = true;
    while (got_better) {
        got_better = false;
        double score = applyBestCandidate(ann_network, type, typesBySpeed, best_score, bestNetworkData, false, silent, 0, best_max_distance);
        if (score < old_score) {
            got_better = true;
            old_score = score;
        }
    }

    optimizeAllNonTopology(ann_network);
    check_score_improvement(ann_network, best_score, bestNetworkData);

    bool old_no_prefiltering = ann_network.options.no_prefiltering;
    ann_network.options.no_prefiltering = true;

    // step 3: slow iterations mode, with increasing max distance
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "\nstep 3: slow iterations mode, with increasing max distance\n";
    }
    int min_dist = 0;
    int max_dist = step_size;
    if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove || type == MoveType::DeltaMinusMove) {
        max_dist = ann_network.options.max_rearrangement_distance;
    }
    got_better = true;
    while (got_better) {
        got_better = false;
        double score = applyBestCandidate(ann_network, type, typesBySpeed, best_score, bestNetworkData, false, silent, min_dist, max_dist);
        if (score < old_score) {
            got_better = true;
            old_score = score;
            min_dist = 0;
            max_dist = step_size;
            if (type == MoveType::RNNIMove || type == MoveType::ArcRemovalMove || type == MoveType::DeltaMinusMove) {
                max_dist = ann_network.options.max_rearrangement_distance;
            }
        } else if (max_dist < ann_network.options.max_rearrangement_distance) {
            got_better = true;
            min_dist = max_dist + 1;
            max_dist = std::min(max_dist + step_size, ann_network.options.max_rearrangement_distance);
        }
    }
    ann_network.options.no_prefiltering = old_no_prefiltering;

    optimizeAllNonTopology(ann_network, true);
    old_score = scoreNetwork(ann_network);
    check_score_improvement(ann_network, best_score, bestNetworkData);

    return old_score;
}

}
