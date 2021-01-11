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

void run_single_start_waves(NetraxOptions& netraxOptions, std::mt19937& rng) {
    std::vector<MoveType> typesBySpeed = {MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove};

    std::vector<double> best_score_by_reticulations(netraxOptions.max_reticulations + 1, std::numeric_limits<double>::infinity());

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_score = std::numeric_limits<double>::infinity();
    int best_num_reticulations = 0;
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);

    std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";
    NetraxInstance::optimizeModel(ann_network);
    NetraxInstance::optimizeBranches(ann_network);
    NetraxInstance::updateReticulationProbs(ann_network);
    std::cout << "Initial network after brlenopt+modelopt+reticulation opt is:\n" << toExtendedNewick(ann_network) << "\n\n";
    std::string best_network = toExtendedNewick(ann_network);
    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << NetraxInstance::scoreNetwork(ann_network) << "\n";

    bool seen_improvement = true;
    while (seen_improvement) {
        seen_improvement = false;

        double new_score = NetraxInstance::optimizeEverythingRun(ann_network, typesBySpeed, start_time);
        std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
        std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";

        if (new_score < best_score) {
            best_score = new_score;
            best_score_by_reticulations[ann_network.network.num_reticulations()] = new_score;
            best_num_reticulations = ann_network.network.num_reticulations();
            std::cout << "IMPROVED BEST SCORE FOUND SO FAR: " << best_score << "\n\n";
            NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
            best_network = toExtendedNewick(ann_network);
            std::cout << best_network << "\n";
            std::cout << "Better network written to " << netraxOptions.output_file << "\n";

            //print displayed trees
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

            best_score_by_reticulations[ann_network.network.num_reticulations()] = new_score;
        } else {
            std::cout << "REMAINED BEST SCORE FOUND SO FAR: " << best_score << "\n";
        }

        if (new_score <= best_score) { // score did not get worse
            if (ann_network.network.num_reticulations() < ann_network.options.max_reticulations) {
                seen_improvement = true;

                // old and deprecated: randomly add new reticulation
                //NetraxInstance::add_extra_reticulations(ann_network, ann_network.network.num_reticulations() + 1);

                // new version: search for best place to add the new reticulation
                MoveType insertionType = MoveType::ArcInsertionMove;
                netrax::greedyHillClimbingTopology(ann_network, insertionType, true, 1);

                NetraxInstance::optimizeModel(ann_network);
                NetraxInstance::optimizeBranches(ann_network);
                NetraxInstance::updateReticulationProbs(ann_network);
                new_score = NetraxInstance::scoreNetwork(ann_network);
                best_score_by_reticulations[ann_network.network.num_reticulations()] = new_score;
                std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";
            }
        }

        bool disabledReticulations = false;
        for (size_t i = 0; i < ann_network.reticulation_probs.size(); ++i) {
            if (ann_network.reticulation_probs[i] == 0.0 || ann_network.reticulation_probs[i] == 1.0) {
                disabledReticulations = true;
                break;
            }
        }
        if ((!seen_improvement || disabledReticulations) && ann_network.network.num_reticulations() > 0) { // try removing arcs
            MoveType removalType = MoveType::ArcRemovalMove;
            size_t old_num_reticulations = ann_network.network.num_reticulations();
            double logl_before_removal = NetraxInstance::computeLoglikelihood(ann_network);
            double bic_before_removal = NetraxInstance::scoreNetwork(ann_network);
            netrax::greedyHillClimbingTopology(ann_network, removalType);
            double logl_after_removal = NetraxInstance::computeLoglikelihood(ann_network);
            double bic_after_removal = NetraxInstance::scoreNetwork(ann_network);
            std::cout << "logl_before_removal: " << logl_before_removal << ", bic_before_removal: " << bic_before_removal << "\n";
            std::cout << "logl_after_removal: " << logl_after_removal << ", bic_after_removal: " << bic_after_removal << "\n";

            if (disabledReticulations) {
                if (ann_network.network.num_reticulations() == old_num_reticulations) {
                    for (size_t i = 0; i < ann_network.network.reticulation_nodes.size(); ++i) {
                        std::cout << "reticulation " << i << " has prob " << ann_network.reticulation_probs[i] << "\n";
                        if (ann_network.reticulation_probs[i] == 1.0 || ann_network.reticulation_probs[i] == 0.0) {
                            ArcRemovalMove removalMove;
                            std::vector<ArcRemovalMove> removalMoves = possibleArcRemovalMoves(ann_network, ann_network.network.reticulation_nodes[i]);
                            size_t wanted_u_clv_index; // clv index of the parent we want to remove
                            if (ann_network.reticulation_probs[i] == 0.0) { // remove edge to first parent
                                wanted_u_clv_index = getReticulationFirstParent(ann_network.network, ann_network.network.reticulation_nodes[i])->clv_index;
                            } else {   
                                wanted_u_clv_index = getReticulationSecondParent(ann_network.network, ann_network.network.reticulation_nodes[i])->clv_index;
                            }
                            for (size_t j = 0; j < removalMoves.size(); ++j) {
                               if (removalMoves[j].u_clv_index == wanted_u_clv_index) {
                                   removalMove = removalMoves[j];
                                   break;
                               }
                            }

                            std::cout << "Chosen removal move: " << toString(removalMove) << "\n";

                            double logl_before = NetraxInstance::computeLoglikelihood(ann_network);
                            double bic_before = NetraxInstance::scoreNetwork(ann_network);

                            std::cout << "Network before move:\n";
                            std::cout << toExtendedNewick(ann_network) << "\n";
                            std::cout << "logl directly before move: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "bic directly before move: " << NetraxInstance::scoreNetwork(ann_network) << "\n";
                            NetraxInstance::optimizeModel(ann_network);
                            std::cout << "logl directly before move, after modelopt: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "bic directly before move, after modelopt: " << NetraxInstance::scoreNetwork(ann_network) << "\n";
                            NetraxInstance::optimizeBranches(ann_network);
                            NetraxInstance::updateReticulationProbs(ann_network);
                            std::cout << "Network before, after modelopt+brlenopt+reticulation opt:\n";
                            std::cout << toExtendedNewick(ann_network) << "\n";
                            std::cout << "logl directly before move, after modelopt+brlenopt+reticulation opt: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "bic directly before move, after modelopt+brlenopt+reticulation opt: " << NetraxInstance::scoreNetwork(ann_network) << "\n";


                            std::cout << "Network after move:\n";
                            netrax::performMove(ann_network, removalMove);
                            std::cout << toExtendedNewick(ann_network) << "\n";
                            std::cout << "logl directly after move: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "bic directly after move: " << NetraxInstance::scoreNetwork(ann_network) << "\n";
                            NetraxInstance::optimizeModel(ann_network);
                            std::cout << "logl directly after move + modelopt: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "bic directly after move + modelopt: " << NetraxInstance::scoreNetwork(ann_network) << "\n";
                            NetraxInstance::optimizeBranches(ann_network);
                            NetraxInstance::updateReticulationProbs(ann_network);
                            std::cout << "logl directly after move + modelopt+brlenopt+reticulation opt: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "bic directly after move + modelopt+brlenopt+reticulation opt: " << NetraxInstance::scoreNetwork(ann_network) << "\n";
                            std::cout << "logl before output: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            double logl_before_output = NetraxInstance::computeLoglikelihood(ann_network);
                            std::cout << toExtendedNewick(ann_network) << "\n";
                            double logl_after_output = NetraxInstance::computeLoglikelihood(ann_network);
                            assert(logl_before_output == logl_after_output);
                            std::cout << "calling logl computation again: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "Calling bic again " << NetraxInstance::scoreNetwork(ann_network) << "\n";
                            std::cout << "printing the network again:\n";
                            std::cout << toExtendedNewick(ann_network) << "\n";
                            std::cout << "calling logl computation again: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                            std::cout << "Calling bic again " << NetraxInstance::scoreNetwork(ann_network) << "\n";

                            double logl_after = NetraxInstance::computeLoglikelihood(ann_network);
                            double bic_after = NetraxInstance::scoreNetwork(ann_network);
                            std::cout << "logl_before: " << logl_before << ", bic_before: " << bic_before << "\n";
                            std::cout << "logl_after: " << logl_after << ", bic_after: " << bic_after << "\n";
                            break;
                        }
                    }
                    assert(false);
                }
                assert(ann_network.network.num_reticulations() < old_num_reticulations);
            }
            if (ann_network.network.num_reticulations() < old_num_reticulations) {
                NetraxInstance::optimizeModel(ann_network);
                NetraxInstance::optimizeBranches(ann_network);
                NetraxInstance::updateReticulationProbs(ann_network);
                new_score = NetraxInstance::scoreNetwork(ann_network);
                if (new_score < best_score_by_reticulations[ann_network.network.num_reticulations()]) {
                    best_score_by_reticulations[ann_network.network.num_reticulations()] = new_score;
                    seen_improvement = true;
                    std::cout << "Got better by removing arcs\n";
                    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";
                }
            }
        }
    }

    std::cout << "The inferred network has " << best_num_reticulations << " reticulations and this BIC score: " << best_score << "\n\n";
    std::cout << "Best found network is:\n" << best_network << "\n\n";

    std::cout << "Statistics on which moves were taken:\n";
    for (const auto& entry : ann_network.stats.moves_taken) {
        std::cout << toString(entry.first) << ": " << entry.second << "\n";
    }
}

void run_single_start(NetraxOptions& netraxOptions, std::mt19937& rng) {
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
    double best_score = std::numeric_limits<double>::infinity();
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
        NetraxInstance::optimizeEverythingInWaves(ann_network);
        double final_bic = NetraxInstance::scoreNetwork(ann_network);
        std::cout << "The inferred network has " << ann_network.network.num_reticulations() << " reticulations and this BIC score: " << final_bic << "\n\n";
        if (final_bic < best_score) {
            best_score = final_bic;
            std::cout << "IMPROVED BEST SCORE FOUND SO FAR: " << best_score << "\n\n";
            NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
            std::cout << "Better network written to " << netraxOptions.output_file << "\n";  
        } else {
            std::cout << "REMAINED BEST SCORE FOUND SO FAR: " << best_score << "\n";
        }
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
        NetraxInstance::optimizeEverythingInWaves(ann_network);
        double final_bic = NetraxInstance::scoreNetwork(ann_network);
        std::cout << "The inferred network has " << ann_network.network.num_reticulations() << " reticulations and this BIC score: " << final_bic << "\n\n";
        if (final_bic < best_score) {
            best_score = final_bic;
            std::cout << "IMPROVED BEST SCORE FOUND SO FAR: " << best_score << "\n\n";
            NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
            std::cout << "Better network written to " << netraxOptions.output_file << "\n";  
        } else {
            std::cout << "REMAINED BEST SCORE FOUND SO FAR: " << best_score << "\n";
        }
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
