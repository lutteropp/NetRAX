#include "NetworkSearch.hpp"
#include <omp.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <random>
#include <raxml-ng/main.hpp>

//#define _RAXML_PTHREADS

#include "SimulatedAnnealing.hpp"
#include "Wavesearch.hpp"

#include "../NetworkDistances.hpp"
#include "../graph/NodeDisplayedTreeData.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../optimization/Optimization.hpp"

namespace netrax {

void judgeNetwork(BestNetworkData &best_network_data,
                  NetraxOptions &netraxOptions, const RaxmlInstance &instance,
                  std::mt19937 &rng) {
  AnnotatedNetwork inferredNetwork = build_annotated_network_from_string(
      netraxOptions, instance,
      best_network_data.newick[best_network_data.best_n_reticulations]);
  init_annotated_network(inferredNetwork, rng);
  AnnotatedNetwork trueNetwork = build_annotated_network_from_file(
      netraxOptions, instance, netraxOptions.true_network_path);
  init_annotated_network(trueNetwork, rng);

  if (inferredNetwork.network.num_tips() != trueNetwork.network.num_tips()) {
    throw std::runtime_error("Unequal number of taxa");
  }

  optimizeAllNonTopology(inferredNetwork, OptimizeAllNonTopologyType::SLOW);
  optimizeAllNonTopology(trueNetwork, OptimizeAllNonTopologyType::SLOW);

  double bic_inferred = scoreNetwork(inferredNetwork);
  double bic_true = scoreNetwork(trueNetwork);

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << "\nEvaluation of inference results:\n";
    std::cout << "bic_inferred: " << bic_inferred << "\n";
    std::cout << "bic_true: " << bic_true << "\n";
    if (bic_inferred < bic_true) {
      std::cout << "Inferred a better BIC.\n";
    } else if (bic_inferred > bic_true) {
      std::cout << "Inferred a worse BIC.\n";
    } else {
      std::cout << "Inferred an equal BIC.\n";
    }

    std::cout << "n_reticulations inferred: "
              << inferredNetwork.network.num_reticulations() << "\n";
    std::cout << "n_reticulations true: "
              << trueNetwork.network.num_reticulations() << "\n";
    if (inferredNetwork.network.num_reticulations() <
        trueNetwork.network.num_reticulations()) {
      std::cout << "Inferred less reticulations.\n";
    } else if (inferredNetwork.network.num_reticulations() >
               trueNetwork.network.num_reticulations()) {
      std::cout << "Inferred more reticulations.\n";
    } else {
      std::cout << "Inferred equal number of reticulations.\n";
    }
  }

  std::unordered_map<std::string, unsigned int> label_to_int;
  for (size_t i = 0; i < inferredNetwork.network.num_tips(); ++i) {
    label_to_int[inferredNetwork.network.nodes_by_index[i]->label] = i;
  }

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << "Unrooted softwired network distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::UNROOTED_SOFTWIRED_DISTANCE)
              << "\n";
    std::cout << "Unrooted hardwired network distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::UNROOTED_HARDWIRED_DISTANCE)
              << "\n";
    std::cout << "Unrooted displayed trees distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::UNROOTED_DISPLAYED_TREES_DISTANCE)
              << "\n";

    std::cout << "Rooted softwired network distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::ROOTED_SOFTWIRED_DISTANCE)
              << "\n";
    std::cout << "Rooted hardwired network distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::ROOTED_HARDWIRED_DISTANCE)
              << "\n";
    std::cout << "Rooted displayed trees distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::ROOTED_DISPLAYED_TREES_DISTANCE)
              << "\n";
    std::cout << "Rooted tripartition distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::ROOTED_TRIPARTITION_DISTANCE)
              << "\n";
    std::cout << "Rooted path multiplicity distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::ROOTED_PATH_MULTIPLICITY_DISTANCE)
              << "\n";
    std::cout << "Rooted nested labels distance: "
              << get_network_distance(
                     inferredNetwork, trueNetwork, label_to_int,
                     NetworkDistanceType::ROOTED_NESTED_LABELS_DISTANCE)
              << "\n";
  }
}

void run_single_start_waves(NetraxOptions &netraxOptions,
                            const RaxmlInstance &instance,
                            const std::vector<MoveType> &typesBySpeed,
                            const std::vector<MoveType> &typesBySpeedGoodStart,
                            std::mt19937 &rng) {
  /* non-master ranks load starting trees from a file */
  ParallelContext::global_mpi_barrier();
  netrax::AnnotatedNetwork ann_network =
      build_annotated_network(netraxOptions, instance);
  init_annotated_network(ann_network, rng);
  BestNetworkData bestNetworkData(ann_network.options.max_reticulations);

  if (hasBadReticulation(ann_network)) {
    throw std::runtime_error(
        "The user-specified start network has a reticulation with 0/1 prob");
  }

  wavesearch(ann_network, &bestNetworkData, typesBySpeed,
             typesBySpeedGoodStart);

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << "Statistics on which moves were taken:\n";
    std::unordered_set<MoveType> seen;
    for (const MoveType &type : typesBySpeed) {
      if (seen.count(type) == 0) {
        std::cout << toString(type) << ": "
                  << ann_network.stats.moves_taken[type] << "\n";
      }
      seen.emplace(type);
    }
    std::cout << "Best inferred network has "
              << bestNetworkData.best_n_reticulations
              << " reticulations, logl = "
              << bestNetworkData.logl[bestNetworkData.best_n_reticulations]
              << ", bic = "
              << bestNetworkData.bic[bestNetworkData.best_n_reticulations]
              << "\n";
    std::cout << "Best inferred network is: \n";
    std::cout << bestNetworkData.newick[bestNetworkData.best_n_reticulations]
              << "\n";

    std::cout << "n_reticulations, logl, bic, newick\n";
    for (size_t i = 0; i < bestNetworkData.bic.size(); ++i) {
      if (bestNetworkData.bic[i] == std::numeric_limits<double>::infinity()) {
        continue;
      }
      std::cout << i << ", " << bestNetworkData.logl[i] << ", "
                << bestNetworkData.bic[i] << ", " << bestNetworkData.newick[i]
                << "\n";

      std::ofstream outfile(ann_network.options.output_file + "_" +
                            std::to_string(i) + "_reticulations.nw");
      outfile << bestNetworkData.newick[i] << "\n";
      outfile.close();
    }
    std::ofstream outfile(ann_network.options.output_file);
    outfile << bestNetworkData.newick[bestNetworkData.best_n_reticulations]
            << "\n";
    outfile.close();
  }

  if (!netraxOptions.true_network_path.empty()) {
    judgeNetwork(bestNetworkData, netraxOptions, instance, rng);
  }
}

void run_random(NetraxOptions &netraxOptions, const RaxmlInstance &instance,
                const std::vector<MoveType> &typesBySpeed, std::mt19937 &rng) {
  std::uniform_int_distribution<long> dist(0, RAND_MAX);
  BestNetworkData bestNetworkData(netraxOptions.max_reticulations);

  Statistics totalStats;
  std::vector<MoveType> allTypes = {
      MoveType::RNNIMove,         MoveType::RSPR1Move,
      MoveType::HeadMove,         MoveType::TailMove,
      MoveType::RSPRMove,         MoveType::DeltaPlusMove,
      MoveType::ArcInsertionMove, MoveType::DeltaMinusMove,
      MoveType::ArcRemovalMove};
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
        std::cout << "Starting with new random network " << n_iterations
                  << " with " << start_reticulations
                  << " reticulations, tree seed = " << seed << ".\n";
      }
      netrax::AnnotatedNetwork ann_network =
          build_random_annotated_network(netraxOptions, instance, seed);
      init_annotated_network(ann_network, rng);
      add_extra_reticulations(ann_network, start_reticulations);

      wavesearch(ann_network, &bestNetworkData, typesBySpeed, typesBySpeed);
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << " Inferred " << ann_network.network.num_reticulations()
                  << " reticulations, logl = "
                  << computeLoglikelihood(ann_network)
                  << ", bic = " << scoreNetwork(ann_network) << "\n";
      }
      for (MoveType type : allTypes) {
        totalStats.moves_taken[type] += ann_network.stats.moves_taken[type];
      }
      // std::cout << "Ending with new random tree with " <<
      // ann_network.network.num_reticulations() << " reticulations.\n";
      if (netraxOptions.timeout > 0) {
        auto act_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(act_time -
                                                             start_time)
                .count() >= netraxOptions.timeout) {
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
        std::cout << "Starting with new parsimony tree " << n_iterations
                  << " with " << start_reticulations
                  << " reticulations, tree seed = " << seed << ".\n";
      }
      netrax::AnnotatedNetwork ann_network =
          build_parsimony_annotated_network(netraxOptions, instance, seed);
      init_annotated_network(ann_network, rng);
      add_extra_reticulations(ann_network, start_reticulations);
      wavesearch(ann_network, &bestNetworkData, typesBySpeed, typesBySpeed);
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << " Inferred " << ann_network.network.num_reticulations()
                  << " reticulations, logl = "
                  << computeLoglikelihood(ann_network)
                  << ", bic = " << scoreNetwork(ann_network) << "\n";
      }
      for (MoveType type : allTypes) {
        totalStats.moves_taken[type] += ann_network.stats.moves_taken[type];
      }
      // std::cout << "Ending with new parsimony tree with " <<
      // ann_network.network.num_reticulations() << " reticulations.\n";
      if (netraxOptions.timeout > 0) {
        auto act_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(act_time -
                                                             start_time)
                .count() >= netraxOptions.timeout) {
          break;
        }
      } else if (n_iterations >= netraxOptions.num_parsimony_start_networks) {
        break;
      }
    }
  }

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    std::cout << "\nAggregated statistics on which moves were taken:\n";
    std::unordered_set<MoveType> seen;
    for (const MoveType &type : typesBySpeed) {
      if (seen.count(type) == 0) {
        std::cout << toString(type) << ": " << totalStats.moves_taken[type]
                  << "\n";
      }
      seen.emplace(type);
    }
    std::cout << "\n";

    std::cout << "Best inferred network has "
              << bestNetworkData.best_n_reticulations
              << " reticulations, logl = "
              << bestNetworkData.logl[bestNetworkData.best_n_reticulations]
              << ", bic = "
              << bestNetworkData.bic[bestNetworkData.best_n_reticulations]
              << "\n";
    std::cout << "Best inferred network is: \n";
    std::cout << bestNetworkData.newick[bestNetworkData.best_n_reticulations]
              << "\n";

    std::cout << "n_reticulations, logl, bic, newick\n";
    for (size_t i = 0; i < bestNetworkData.bic.size(); ++i) {
      if (bestNetworkData.bic[i] == std::numeric_limits<double>::infinity()) {
        continue;
      }
      std::cout << i << ", " << bestNetworkData.logl[i] << ", "
                << bestNetworkData.bic[i] << ", " << bestNetworkData.newick[i]
                << "\n";

      std::ofstream outfile(netraxOptions.output_file + "_" +
                            std::to_string(i) + "_reticulations.nw");
      outfile << bestNetworkData.newick[i] << "\n";
      outfile.close();
    }
    std::ofstream outfile(netraxOptions.output_file);
    outfile << bestNetworkData.newick[bestNetworkData.best_n_reticulations]
            << "\n";
    outfile.close();
  }

  if (!netraxOptions.true_network_path.empty()) {
    judgeNetwork(bestNetworkData, netraxOptions, instance, rng);
  }
}

}  // namespace netrax
