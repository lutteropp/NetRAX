/*
 * Api.cpp
 * 
 * Api functions for using NetRAX.
 *
 *  Created on: Apr 24, 2020
 *      Author: Sarah Lutteropp
 */

#include "Api.hpp"

#include <stddef.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

#include <raxml-ng/TreeInfo.hpp>
#include <raxml-ng/main.hpp>
#include "graph/AnnotatedNetwork.hpp"
#include "graph/BiconnectedComponents.hpp"
#include "graph/Network.hpp"
#include "graph/NetworkFunctions.hpp"
#include "graph/NetworkTopology.hpp"
#include "io/NetworkIO.hpp"
#include "likelihood/LikelihoodComputation.hpp"
#include "NetraxOptions.hpp"
#include "RaxmlWrapper.hpp"
#include "optimization/Moves.hpp"
#include "optimization/BranchLengthOptimization.hpp"
#include "optimization/TopologyOptimization.hpp"
#include "DebugPrintFunctions.hpp"
#include "utils.hpp"

namespace netrax {

void allocateBranchProbsArray(AnnotatedNetwork& ann_network) {
    // allocate branch probs array...
     ann_network.reticulation_probs = std::vector<double>(ann_network.options.max_reticulations, 0.5);
}

/**
 * Initializes the network annotations. Precomputes the reversed topological sort, the partitioning into blobs, and sets the branch probabilities.
 * 
 * @param ann_network The still uninitialized annotated network.
 */
void NetraxInstance::init_annotated_network(AnnotatedNetwork &ann_network, std::mt19937& rng) {
    Network &network = ann_network.network;
    ann_network.rng = rng;

    ann_network.travbuffer = netrax::reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = netrax::partitionNetworkIntoBlobs(network, ann_network.travbuffer);

    allocateBranchProbsArray(ann_network);
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        double firstParentProb = ann_network.network.edges_by_index[ann_network.network.reticulation_nodes[i]->getReticulationData()->getLinkToFirstParent()->edge_pmatrix_index]->prob;
        ann_network.reticulation_probs[i] = firstParentProb;
    }

    netrax::RaxmlWrapper wrapper(ann_network.options);
    ann_network.raxml_treeinfo = std::unique_ptr<TreeInfo>(wrapper.createRaxmlTreeinfo(ann_network));

    assert(static_cast<RaxmlWrapper::NetworkParams*>(ann_network.raxml_treeinfo->pll_treeinfo().likelihood_computation_params)->ann_network == &ann_network);

    ann_network.fake_treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    netrax::setup_pmatrices(ann_network, false, true);
    for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
        ann_network.fake_treeinfo->clv_valid[i][network.root->clv_index] = 0;
    }
    netrax::computeLoglikelihood(ann_network, false, true);
}

void NetraxInstance::init_annotated_network(AnnotatedNetwork &ann_network) {
    std::random_device dev;
    std::mt19937 rng(dev());
    init_annotated_network(ann_network, rng);
}

/**
 * Creates the inital annotated network data structure to work on, reading the network from the file specified in the options.
 * 
 * @param options The information specified by the user.
 */
AnnotatedNetwork NetraxInstance::build_annotated_network(NetraxOptions &options) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = std::move(netrax::readNetworkFromFile(options.start_network_file,
            options.max_reticulations));
    return ann_network;
}

/**
 * Creates the inital annotated network data structure to work on, reading the network from a given string and ignoring the input file specified in the options.
 * 
 * @param options The information specified by the user.
 * @param newickString The network in Extended Newick format.
 */
AnnotatedNetwork NetraxInstance::build_annotated_network_from_string(NetraxOptions &options,
        const std::string &newickString) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = netrax::readNetworkFromString(newickString, options.max_reticulations);
    return ann_network;
}

/**
 * Converts a pll_utree_t into an annotated network.
 * 
 * @param options The options specified by the user.
 * @param utree The pll_utree_t to be converted into an annotated network.
 */
AnnotatedNetwork NetraxInstance::build_annotated_network_from_utree(NetraxOptions &options,
        const pll_utree_t &utree) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = netrax::convertUtreeToNetwork(utree, options.max_reticulations);
    return ann_network;
}

/**
 * Takes a network and adds random reticulations to it, until the specified targetCount has been reached.
 * 
 * @param ann_network The network we want to add reticulations to.
 * @param targetCount The number of reticulations we want to have in the network.
 */
void NetraxInstance::add_extra_reticulations(AnnotatedNetwork &ann_network, unsigned int targetCount) {
    if (targetCount > ann_network.options.max_reticulations) {
        throw std::runtime_error("Please increase maximum allowed number of reticulations");
    }

    Network &network = ann_network.network;
    std::mt19937 &rng = ann_network.rng;

    if (targetCount < network.num_reticulations()) {
        throw std::invalid_argument("The target count is smaller than the current number of reticulations in the network");
    }

    // TODO: This can be implemented more eficiently than by always re-gathering all candidates
    while (targetCount > network.num_reticulations()) {
        std::vector<ArcInsertionMove> candidates = possibleArcInsertionMoves(ann_network);
        assert(!candidates.empty());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, candidates.size() - 1);
        ArcInsertionMove move = candidates[dist(rng)];
        performMove(ann_network, move);
    }
}

/**
 * Creates a random annotated network.
 * 
 * Creates an annotated network by building a random tree..
 * 
 * @param options The options specified by the user.
 */
AnnotatedNetwork NetraxInstance::build_random_annotated_network(NetraxOptions &options) {
    RaxmlWrapper wrapper(options);
    Tree tree = wrapper.generateRandomTree();
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, tree.pll_utree());
    return ann_network;
}

/**
 * Creates an annotated network by building a maximum parsimony tree..
 * 
 * @param options The options specified by the user.
 */
AnnotatedNetwork NetraxInstance::build_parsimony_annotated_network(NetraxOptions &options) {
    RaxmlWrapper wrapper(options);
    Tree tree = wrapper.generateParsimonyTree();
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, tree.pll_utree());
    return ann_network;
}

/**
 * Creates an annotated network by building a maximum likelihood tree..
 * 
 * @param options The options specified by the user.
 */
AnnotatedNetwork NetraxInstance::build_best_raxml_annotated_network(NetraxOptions &options) {
    RaxmlWrapper wrapper(options);
    Tree tree = wrapper.bestRaxmlTree();
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, tree.pll_utree());
    return ann_network;
}

/**
 * Computes the loglikelihood of a given network.
 * 
 * @param ann_network The network.
 */
double NetraxInstance::computeLoglikelihood(AnnotatedNetwork &ann_network) {
    double logl = ann_network.raxml_treeinfo->loglh(true);
    return logl;
}

/**
 * Computes the BIC-score of a given network. A smaller score is a better score.
 * 
 * @param ann_network The network.
 */
double NetraxInstance::scoreNetwork(AnnotatedNetwork &ann_network) {
    double logl = computeLoglikelihood(ann_network);
    return bic(ann_network, logl);
}

/**
 * Re-infers the reticulation probabilities of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::updateReticulationProbs(AnnotatedNetwork &ann_network) {
    if (ann_network.network.num_reticulations() == 0) {
        return;
    }
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    netrax::optimize_reticulations(ann_network, 100);
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after updating reticulation probs: " << new_score << "\n";
    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

/**
 * Re-infers the likelihood model parameters of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeModel(AnnotatedNetwork &ann_network) {
    double incremental_logl = netrax::computeLoglikelihood(ann_network, 1, 1);
    double non_incremental_logl = netrax::computeLoglikelihood(ann_network, 0, 1);
    if (incremental_logl != non_incremental_logl) {
        std::cout << "incremental logl: " << incremental_logl << "\n";
        std::cout << "non_incremental logl: " << non_incremental_logl << "\n";
    }
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    std::cout << "BIC score before model optimization: " << old_score << "\n";
    ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after model optimization: " << new_score << "\n";
    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

/**
 * Re-infers the branch lengths of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeBranches(AnnotatedNetwork &ann_network) {
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    ann_network.raxml_treeinfo->optimize_branches(ann_network.options.lh_epsilon, 1);
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after branch length optimization: " << new_score << "\n";
    assert(new_score <= old_score + ann_network.options.score_epsilon);
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        old_score = scoreNetwork(ann_network);
        pllmod_algo_opt_brlen_scalers_treeinfo(ann_network.fake_treeinfo,
                                                        RAXML_BRLEN_SCALER_MIN,
                                                        RAXML_BRLEN_SCALER_MAX,
                                                        ann_network.options.brlen_min,
                                                        ann_network.options.brlen_max,
                                                        RAXML_PARAM_EPSILON);
        new_score = scoreNetwork(ann_network);
        std::cout << "BIC score after branch length scaler optimization: " << new_score << "\n";
        assert(new_score <= old_score + ann_network.options.score_epsilon);
    }
}

/**
 * Re-infers the topology of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types) {
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    greedyHillClimbingTopology(ann_network, types);
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC after topology optimization: " << new_score << "\n";

    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

/**
 * Re-infers the topology of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeTopology(AnnotatedNetwork &ann_network, MoveType& type) {
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    greedyHillClimbingTopology(ann_network, type);
    double new_score = scoreNetwork(ann_network);
    //std::cout << "BIC after topology optimization: " << new_score << "\n";

    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

double NetraxInstance::optimizeEverythingRun(AnnotatedNetwork & ann_network, std::vector<MoveType>& typesBySpeed, const std::chrono::high_resolution_clock::time_point& start_time) {
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
        //std::cout << "Using move type: " << toString(typesBySpeed[type_idx]) << "\n";
        double old_score = scoreNetwork(ann_network);
        optimizeTopology(ann_network, typesBySpeed[type_idx]);
        double new_score = scoreNetwork(ann_network);
        if (old_score - new_score > ann_network.options.score_epsilon) { // score got better
            //std::cout << "BIC after topology optimization: " << new_score << "\n";
            //std::cout << "Current number of reticulations: " << ann_network.network.num_reticulations() << "\n";
            //std::cout << "network (BIC = " << new_score << ", logl = " << computeLoglikelihood(ann_network) << ") before brlen opt:\n" << toExtendedNewick(ann_network.network) << "\n";
            //optimizeBranches(ann_network);
            //new_score = scoreNetwork(ann_network);
            //std::cout << "network (BIC = " << new_score << ", logl = " << computeLoglikelihood(ann_network) << ") after brlen opt:\n" << toExtendedNewick(ann_network.network) << "\n\n";
            //updateReticulationProbs(ann_network);
            //optimizeModel(ann_network);
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

    NetraxInstance::optimizeAllNonTopology(ann_network, true);
    best_score = scoreNetwork(ann_network);

    return best_score;
}

/**
 * Re-infers everything (brach lengths, reticulation probabilities, likelihood model parameters, topology).
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeEverything(AnnotatedNetwork &ann_network) {
    std::vector<MoveType> typesBySpeed = {MoveType::DeltaMinusMove, MoveType::ArcRemovalMove, MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove, MoveType::DeltaPlusMove, MoveType::ArcInsertionMove};
    auto start_time = std::chrono::high_resolution_clock::now();

    optimizeBranches(ann_network);
    updateReticulationProbs(ann_network);
    optimizeModel(ann_network);
    double new_score = scoreNetwork(ann_network);
    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << computeLoglikelihood(ann_network) << "\n";
    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";
    //assert(new_score <= initial_score);

    //double_check_likelihood(ann_network);

    optimizeEverythingRun(ann_network, typesBySpeed, start_time);

    std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << computeLoglikelihood(ann_network) << "\n";
    std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";

    std::cout << "Statistics on which moves were taken:\n";
    for (const auto& entry : ann_network.stats.moves_taken) {
        std::cout << toString(entry.first) << ": " << entry.second << "\n";
    }
}

/**
 * Re-infers everything (brach lengths, reticulation probabilities, likelihood model parameters, topology).
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeEverythingInWaves(AnnotatedNetwork &ann_network) {
    throw std::runtime_error("this is deprecated code");
    std::vector<MoveType> typesBySpeed = {MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove};
    auto start_time = std::chrono::high_resolution_clock::now();

    optimizeBranches(ann_network);
    updateReticulationProbs(ann_network);
    optimizeModel(ann_network);
    double new_score = scoreNetwork(ann_network);
    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << computeLoglikelihood(ann_network) << "\n";
    std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";
    //assert(new_score <= initial_score);

    //double_check_likelihood(ann_network);

    double best_score = new_score;
    bool seen_improvement = true;

    while (seen_improvement) {
        seen_improvement = false;

        new_score = optimizeEverythingRun(ann_network, typesBySpeed, start_time);
        std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << computeLoglikelihood(ann_network) << "\n";
        std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";

        if (new_score <= best_score) { // score did not get worse
            best_score = new_score;
            if (ann_network.network.num_reticulations() < ann_network.options.max_reticulations) {
                seen_improvement = true;
                add_extra_reticulations(ann_network, 1);
                optimizeBranches(ann_network);
                updateReticulationProbs(ann_network);
                optimizeModel(ann_network);
                new_score = scoreNetwork(ann_network);
                std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << computeLoglikelihood(ann_network) << "\n";
                std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";
            }
        }
    }

    std::cout << "Statistics on which moves were taken:\n";
    for (const auto& entry : ann_network.stats.moves_taken) {
        std::cout << toString(entry.first) << ": " << entry.second << "\n";
    }
}


/**
 * Writes a network to a file in Extended Newick Format.
 * 
 * @param ann_network The network.
 * @param filepath The file where to write the network to.
 */
void NetraxInstance::writeNetwork(AnnotatedNetwork &ann_network, const std::string &filepath) {
    std::ofstream outfile(filepath);
    outfile << netrax::toExtendedNewick(ann_network) << "\n";
    outfile.close();
}

void NetraxInstance::double_check_likelihood(AnnotatedNetwork &ann_network) {
    double logl = ann_network.raxml_treeinfo->loglh(true);
    std::string newick = toExtendedNewick(ann_network.network);

    AnnotatedNetwork ann_network2;
    ann_network2.options = ann_network.options;
    ann_network2.network = netrax::readNetworkFromString(newick,
            ann_network.options.max_reticulations);
    init_annotated_network(ann_network2, ann_network.rng);
    NetraxInstance::optimizeModel(ann_network2);
    double reread_logl = NetraxInstance::computeLoglikelihood(ann_network2);

    bool similar_logl = (abs(logl - reread_logl < 1.0));

    if (!similar_logl) {
        std::cout << "logl: " << logl << "\n";
        std::cout << "reread_logl: " << reread_logl << "\n";
        std::cout << "current network:\n" << newick << "\n";
        std::cout << exportDebugInfo(ann_network) << "\n";
        std::cout << "reread_network:\n" << toExtendedNewick(ann_network2) << "\n";
        std::cout << exportDebugInfo(ann_network2) << "\n";
    }

    assert(similar_logl);
}

void NetraxInstance::optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt) {
    bool gotBetter = true;
    while (gotBetter) {
        gotBetter = false;
        double score_before = scoreNetwork(ann_network);
        NetraxInstance::optimizeModel(ann_network);
        NetraxInstance::optimizeBranches(ann_network);
        NetraxInstance::updateReticulationProbs(ann_network);
        double score_after = scoreNetwork(ann_network);

        if (score_after < score_before - ann_network.options.score_epsilon && extremeOpt) {
            gotBetter = true;
        }
    }
}

}
