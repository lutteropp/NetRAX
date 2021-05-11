#include "Wavesearch.hpp"

#include "CandidateSelection.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/PseudoLoglikelihood.hpp"
#include "../moves/Moves.hpp"
#include "../optimization/Optimization.hpp"
#include "Scrambling.hpp"

namespace netrax {

double optimizeEverythingRun(AnnotatedNetwork& ann_network, const std::vector<MoveType>& typesBySpeed, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, const std::chrono::high_resolution_clock::time_point& start_time, BestNetworkData* bestNetworkData, bool silent = true) {
    unsigned int type_idx = 0;
    unsigned int max_seconds = ann_network.options.timeout;
    double best_score = scoreNetwork(ann_network);
    bool rspr1_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(), MoveType::RSPR1Move) != typesBySpeed.end());
    bool delta_plus_present = (std::find(typesBySpeed.begin(), typesBySpeed.end(), MoveType::DeltaPlusMove) != typesBySpeed.end());

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
        double new_score;
        if (ann_network.options.full_search_by_type) {
            new_score = fullSearch(ann_network, typesBySpeed[type_idx], typesBySpeed, &best_score, bestNetworkData, silent);
        } else {
            new_score = applyBestCandidate(ann_network, typesBySpeed[type_idx], typesBySpeed, &best_score, bestNetworkData, false, silent);
        }

        if (new_score < old_score) { // score got better
            new_score = scoreNetwork(ann_network);
            best_score = new_score;

            if (typesBySpeed.size() > 1) {
                type_idx = 0; // go back to fastest move type
            } else if (ann_network.options.full_search_by_type) {
                type_idx++;
            }
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

    //optimizeAllNonTopology(ann_network, true);
    best_score = scoreNetwork(ann_network);

    return best_score;
}

void wavesearch_internal(AnnotatedNetwork& ann_network, BestNetworkData* bestNetworkData, const std::vector<MoveType>& typesBySpeed, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, double* best_score, const std::chrono::high_resolution_clock::time_point& start_time, bool silent = true) {
    double old_best_score = *best_score;
    bool got_better = true;

    check_score_improvement(ann_network, best_score, bestNetworkData);
    optimizeEverythingRun(ann_network, typesBySpeed, start_state_to_reuse, best_state_to_reuse, start_time, bestNetworkData);
    check_score_improvement(ann_network, best_score, bestNetworkData);

    if (ann_network.options.enforce_extra_search) {
        // next, try enforcing some arc insertion
        while (got_better) {
            got_better = false;
            
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << "Enforcing an arc insertion...\n";
            }
            if (!ann_network.options.no_arc_insertion_moves) {
                applyBestCandidate(ann_network, MoveType::ArcInsertionMove, typesBySpeed, best_score, bestNetworkData, true, silent);
            } else {
                applyBestCandidate(ann_network, MoveType::DeltaMinusMove, typesBySpeed, best_score, bestNetworkData, true, silent);
            }
            check_score_improvement(ann_network, best_score, bestNetworkData);
            optimizeEverythingRun(ann_network, typesBySpeed, start_state_to_reuse, best_state_to_reuse, start_time, bestNetworkData);
            check_score_improvement(ann_network, best_score, bestNetworkData);
            if (*best_score < old_best_score) {
                got_better = true;
                old_best_score = *best_score;
            }
        }
    }
}

void wavesearch_main_internal(AnnotatedNetwork& ann_network, BestNetworkData* bestNetworkData, const std::vector<MoveType>& typesBySpeed, NetworkState& start_state_to_reuse, NetworkState& best_state_to_reuse, double* best_score, const std::chrono::high_resolution_clock::time_point& start_time, bool silent = false) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Starting wavesearch with move types: ";
        for (size_t j = 0; j < typesBySpeed.size(); ++j) {
            std::cout << toString(typesBySpeed[j]);
            if (j + 1 < typesBySpeed.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    }

    wavesearch_internal(ann_network, bestNetworkData, typesBySpeed, start_state_to_reuse, best_state_to_reuse, best_score, start_time, silent);

    double old_best_score = *best_score;

    if (ann_network.options.scrambling > 0) {
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
            std::cout << " Starting scrambling phase...\n";
        }
        unsigned int tries = 0;
        NetworkState bestState = extract_network_state(ann_network);
        double old_best_score = *best_score;
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
            if (!silent) std::cout << " Network before scrambling has BIC Score: " << scoreNetwork(ann_network) << "\n";
        }
        while (tries < ann_network.options.scrambling) {
            apply_network_state(ann_network, bestState);
            double old_best_score_scrambling = scoreNetwork(ann_network);
            scrambleNetwork(ann_network, MoveType::RSPRMove, ann_network.options.scrambling_radius);
            bool improved = true;
            while (improved) {
                improved = false;
                wavesearch_internal(ann_network, bestNetworkData, typesBySpeed, start_state_to_reuse, best_state_to_reuse, best_score, start_time, silent);
                if (*best_score < old_best_score_scrambling) {
                    old_best_score_scrambling = *best_score;
                    improved = true;
                }
            }
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                if (!silent) std::cout << " scrambling BIC: " << scoreNetwork(ann_network) << "\n";
            }
            if (*best_score < old_best_score) {
                old_best_score = *best_score;
                extract_network_state(ann_network, bestState);
                tries = 0;
            } else {
                tries++;
            }
        }
        apply_network_state(ann_network, bestState);
    }
}


void wavesearch(AnnotatedNetwork& ann_network, BestNetworkData* bestNetworkData, const std::vector<MoveType>& typesBySpeed, bool silent) {
    NetworkState start_state_to_reuse = extract_network_state(ann_network, false);
    NetworkState best_state_to_reuse = extract_network_state(ann_network, false);
    auto start_time = std::chrono::high_resolution_clock::now();
    double best_score = std::numeric_limits<double>::infinity();
    ScoreImprovementResult score_improvement;

    if (ann_network.options.computePseudo) {
        double pseudo = computePseudoLoglikelihood(ann_network);
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
            std::cout << "pseudo-loglh: " << pseudo << "\n";
            std::cout << "pseudo-bic: " << bic(ann_network, pseudo) << "\n";
        }
    }

    //optimizeAllNonTopology(ann_network, true);
    optimizeAllNonTopology(ann_network);
    score_improvement = check_score_improvement(ann_network, &best_score, bestNetworkData);

    //std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";

    if (!ann_network.options.start_network_file.empty()) { // don't waste time trying to first horizontally optimize the user-given start network
        if (ann_network.options.no_arc_insertion_moves) {
            if (ann_network.options.full_search_by_type) {
                fullSearch(ann_network, MoveType::DeltaPlusMove, typesBySpeed, &best_score, bestNetworkData, silent);
            } else {
                applyBestCandidate(ann_network, MoveType::DeltaPlusMove, typesBySpeed, &best_score, bestNetworkData, false, silent);
            }
        } else {
            if (ann_network.options.full_search_by_type) {
                fullSearch(ann_network, MoveType::ArcInsertionMove, typesBySpeed, &best_score, bestNetworkData, silent);
            } else {
                applyBestCandidate(ann_network, MoveType::ArcInsertionMove, typesBySpeed, &best_score, bestNetworkData, false, silent);
            }
        }
    }

    wavesearch_main_internal(ann_network, bestNetworkData, typesBySpeed, start_state_to_reuse, best_state_to_reuse, &best_score, start_time, silent);
}

}