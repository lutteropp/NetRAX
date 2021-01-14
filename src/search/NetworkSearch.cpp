#include "NetworkSearch.hpp"
#include <iostream>
#include <string>
#include <vector>

#include <mpreal.h>
#include "../Api.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../io/NetworkIO.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../optimization/TopologyOptimization.hpp"
#include "../optimization/Moves.hpp"
#include "../optimization/MoveType.hpp"

namespace netrax {

struct ScoreImprovementResult {
    bool local_improved = false;
    bool global_improved = false;
};

void printDisplayedTrees(AnnotatedNetwork& ann_network) {
    std::vector<std::pair<std::string, double>> displayed_trees;
    if (ann_network.network.num_reticulations() == 0) {
        std::string newick = netrax::toExtendedNewick(ann_network);
        displayed_trees.emplace_back(std::make_pair(newick, 1.0));
    } else {
        for (int tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index) {
            pll_utree_t* utree = netrax::displayed_tree_to_utree(ann_network.network, tree_index);
            double prob = netrax::displayed_tree_prob(ann_network, tree_index);
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

ScoreImprovementResult check_score_improvement(AnnotatedNetwork& ann_network, double* local_best_per_reticulation, double* local_best, double* global_best) {
    bool local_improved = false;
    bool global_improved = false;
    double new_score = NetraxInstance::scoreNetwork(ann_network);
    double score_diff = *local_best_per_reticulation - new_score;
    if (score_diff > 0.0001) {
        *local_best_per_reticulation = new_score;
        local_improved = true;
        if (new_score < *local_best) {
            std::cout << "SCORE DIFF: " << score_diff << "\n";
            std::cout << "OLD LOCAL BEST SCORE WAS: " << *local_best << "\n";
            *local_best = new_score;
            std::cout << "IMPROVED LOCAL BEST SCORE FOUND SO FAR: " << new_score << "\n\n";
            if (*local_best < *global_best) {
                global_improved = true;
                std::cout << "OLD LOCAL BEST SCORE WAS: " << *global_best << "\n";
                *global_best = *local_best;
                std::cout << "IMPROVED GLOBAL BEST SCORE FOUND SO FAR: " << new_score << "\n\n";
                NetraxInstance::writeNetwork(ann_network, ann_network.options.output_file);
                std::cout << toExtendedNewick(ann_network) << "\n";
                std::cout << "Better network written to " << ann_network.options.output_file << "\n";
                printDisplayedTrees(ann_network);
            } else {
                std::cout << "REMAINED GLOBAL BEST SCORE FOUND SO FAR: " << *global_best << "\n\n";
            }
        } else {
            std::cout << "REMAINED LOCAL BEST SCORE FOUND SO FAR: " << *local_best << "\n\n";
        }
    }
    return ScoreImprovementResult{local_improved, global_improved};
}

void wavesearch(AnnotatedNetwork& ann_network, double* global_best, std::mt19937& rng) {
    std::vector<MoveType> typesBySpeed = {MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove};

    std::vector<double> best_score_by_reticulations(ann_network.options.max_reticulations + 1, std::numeric_limits<double>::infinity());

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_score = std::numeric_limits<double>::infinity();

    std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";
    NetraxInstance::optimizeAllNonTopology(ann_network, true);
    std::cout << "Initial network after modelopt+brlenopt+reticulation opt is:\n" << toExtendedNewick(ann_network) << "\n\n";
    std::string best_network = toExtendedNewick(ann_network);
    ScoreImprovementResult score_improvement;

    score_improvement = check_score_improvement(ann_network, &best_score_by_reticulations[ann_network.network.num_reticulations()], &best_score, global_best);

    // try horizontal moves
    NetraxInstance::optimizeEverythingRun(ann_network, typesBySpeed, start_time);
    score_improvement = check_score_improvement(ann_network, &best_score_by_reticulations[ann_network.network.num_reticulations()], &best_score, global_best);

    bool keepSearching = true;
    while (keepSearching) {
        score_improvement = {false, false};
        keepSearching = false;
        std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
        std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << NetraxInstance::scoreNetwork(ann_network) << "\n";

        // try removing reticulations
        if (ann_network.network.num_reticulations() > 0) { // try removing arcs
            MoveType removalType = MoveType::ArcRemovalMove;
            size_t old_num_reticulations = ann_network.network.num_reticulations();
            double logl_before_removal = NetraxInstance::computeLoglikelihood(ann_network);
            double bic_before_removal = NetraxInstance::scoreNetwork(ann_network);
            netrax::greedyHillClimbingTopology(ann_network, removalType, false);
            double logl_after_removal = NetraxInstance::computeLoglikelihood(ann_network);
            double bic_after_removal = NetraxInstance::scoreNetwork(ann_network);
            std::cout << "logl_before_removal: " << logl_before_removal << ", bic_before_removal: " << bic_before_removal << "\n";
            std::cout << "logl_after_removal: " << logl_after_removal << ", bic_after_removal: " << bic_after_removal << "\n";

            if (ann_network.network.num_reticulations() < old_num_reticulations) {
                NetraxInstance::optimizeAllNonTopology(ann_network);
                NetraxInstance::optimizeEverythingRun(ann_network, typesBySpeed, start_time);
                score_improvement = check_score_improvement(ann_network, &best_score_by_reticulations[ann_network.network.num_reticulations()], &best_score, global_best);
            }
            if (score_improvement.local_improved) {
                keepSearching = true;
                continue;
            }
        }

        // then try adding a reticulation
        if (ann_network.network.num_reticulations() < ann_network.options.max_reticulations) {
            // old and deprecated: randomly add new reticulation
            //NetraxInstance::add_extra_reticulations(ann_network, ann_network.network.num_reticulations() + 1);

            // new version: search for best place to add the new reticulation
            MoveType insertionType = MoveType::ArcInsertionMove;
            netrax::greedyHillClimbingTopology(ann_network, insertionType, true, 1);
            NetraxInstance::optimizeAllNonTopology(ann_network);

            NetraxInstance::optimizeEverythingRun(ann_network, typesBySpeed, start_time);
            score_improvement = check_score_improvement(ann_network, &best_score_by_reticulations[ann_network.network.num_reticulations()], &best_score, global_best);
            if (score_improvement.local_improved) {
                keepSearching = true;
                continue;
            }
        }
    }
}

void oldsearch(AnnotatedNetwork& ann_network, double* global_best, std::mt19937& rng) {
}

void run_single_start_waves(NetraxOptions& netraxOptions, std::mt19937& rng) {
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);
    double global_best = std::numeric_limits<double>::infinity();
    wavesearch(ann_network, &global_best, rng);

    std::cout << "Statistics on which moves were taken:\n";
    for (const auto& entry : ann_network.stats.moves_taken) {
        std::cout << toString(entry.first) << ": " << entry.second << "\n";
    }
}

void run_single_start(NetraxOptions& netraxOptions, std::mt19937& rng) {
    throw std::runtime_error("This is deprecated code");
    double best_score = std::numeric_limits<double>::infinity();

    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);

    std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";
    std::string best_network = toExtendedNewick(ann_network);

    NetraxInstance::optimizeEverything(ann_network);
    double final_bic = NetraxInstance::scoreNetwork(ann_network);
    std::cout << "The inferred network has " << ann_network.network.num_reticulations() << " reticulations and this BIC score: " << final_bic << "\n\n";
    if (final_bic < best_score) {
        best_score = final_bic;
        std::cout << "IMPROVED BEST SCORE FOUND SO FAR: " << best_score << "\n\n";
        NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
        best_network = toExtendedNewick(ann_network);
        std::cout << best_network << "\n";
        std::cout << "Better network written to " << netraxOptions.output_file << "\n";  
    } else {
        std::cout << "REMAINED BEST SCORE FOUND SO FAR: " << best_score << "\n";
    }

    std::cout << "Best found network is:\n" << best_network << "\n\n";
    std::cout << "Recomputing BIC on exactly this network gives: " << NetraxInstance::scoreNetwork(ann_network) << "\n";
}

void run_random(NetraxOptions& netraxOptions, std::mt19937& rng) {
    double global_best = std::numeric_limits<double>::infinity();

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t start_reticulations = 0;
    size_t n_iterations = 0;
    // random start networks
    while (true) {
        n_iterations++;
        std::cout << "Starting with new random network with " << start_reticulations << " reticulations.\n";
        netrax::AnnotatedNetwork ann_network = NetraxInstance::build_random_annotated_network(netraxOptions);
        NetraxInstance::init_annotated_network(ann_network, rng);
        NetraxInstance::add_extra_reticulations(ann_network, start_reticulations);

        wavesearch(ann_network, &global_best, rng);
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
        netrax::AnnotatedNetwork ann_network = NetraxInstance::build_parsimony_annotated_network(netraxOptions);
        NetraxInstance::init_annotated_network(ann_network, rng);
        NetraxInstance::add_extra_reticulations(ann_network, start_reticulations);
        wavesearch(ann_network, &global_best, rng);
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

}
