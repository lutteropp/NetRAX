#include "NetworkSearch.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <limits>
#include <omp.h>

#include <random>
#include <raxml-ng/main.hpp>

//#define _RAXML_PTHREADS

#include "SimulatedAnnealing.hpp"
#include "Wavesearch.hpp"

#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"

namespace netrax {

void run_single_start_waves(NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng) {
    /* non-master ranks load starting trees from a file */
    ParallelContext::global_mpi_barrier();
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions, instance);
    init_annotated_network(ann_network, rng);
    BestNetworkData bestNetworkData(ann_network.options.max_reticulations);
    wavesearch(ann_network, &bestNetworkData, typesBySpeed);

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
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

void run_random(NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng) {
    std::uniform_int_distribution<long> dist(0, RAND_MAX);
    BestNetworkData bestNetworkData(netraxOptions.max_reticulations);

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
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << "Starting with new random network " << n_iterations << " with " << start_reticulations << " reticulations, tree seed = " << seed << ".\n";
            }
            netrax::AnnotatedNetwork ann_network = build_random_annotated_network(netraxOptions, instance, seed);
            init_annotated_network(ann_network, rng);
            add_extra_reticulations(ann_network, start_reticulations);

            wavesearch(ann_network, &bestNetworkData, typesBySpeed);
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
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
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << "Starting with new parsimony tree " << n_iterations << " with " << start_reticulations << " reticulations, tree seed = " << seed << ".\n";
            }
            netrax::AnnotatedNetwork ann_network = build_parsimony_annotated_network(netraxOptions, instance, seed);
            init_annotated_network(ann_network, rng);
            add_extra_reticulations(ann_network, start_reticulations);
            wavesearch(ann_network, &bestNetworkData, typesBySpeed);
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
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

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
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
