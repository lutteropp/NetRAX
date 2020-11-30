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
#include "optimization/TopologyOptimization.hpp"
#include "DebugPrintFunctions.hpp"

namespace netrax {

void allocateBranchProbsArray(AnnotatedNetwork& ann_network) {
    // allocate branch probs array...
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) { // common branches
        ann_network.branch_probs = std::vector<std::vector<double> >(1,
                std::vector<double>(ann_network.network.edges.size() + 1, 1.0));
    } else { // each partition has extra branch properties
        ann_network.branch_probs = std::vector<std::vector<double> >(
                ann_network.fake_treeinfo->partition_count,
                std::vector<double>(ann_network.network.edges.size() + 1, 1.0));
    }
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
    for (size_t p = 0; p < ann_network.branch_probs.size(); ++p) {
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            Node *retNode = ann_network.network.reticulation_nodes[i];
            double firstParentProb = netrax::getReticulationFirstParentProb(network, retNode);
            double secondParentProb = netrax::getReticulationSecondParentProb(network, retNode);
            size_t firstParentEdgeIndex = netrax::getReticulationFirstParentPmatrixIndex(retNode);
            size_t secondParentEdgeIndex = netrax::getReticulationSecondParentPmatrixIndex(retNode);
            ann_network.branch_probs[p][firstParentEdgeIndex] = firstParentProb;
            ann_network.branch_probs[p][secondParentEdgeIndex] = secondParentProb;
        }
    }
    netrax::RaxmlWrapper wrapper(ann_network.options);
    ann_network.raxml_treeinfo = std::unique_ptr<TreeInfo>(wrapper.createRaxmlTreeinfo(ann_network));

    assert(static_cast<RaxmlWrapper::NetworkParams*>(ann_network.raxml_treeinfo->pll_treeinfo().likelihood_computation_params)->ann_network == &ann_network);
    netrax::computeLoglikelihood(ann_network, false, true, false);
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
AnnotatedNetwork NetraxInstance::build_annotated_network(const NetraxOptions &options) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = netrax::readNetworkFromFile(options.start_network_file,
            options.max_reticulations);
    return ann_network;
}

/**
 * Creates the inital annotated network data structure to work on, reading the network from a given string and ignoring the input file specified in the options.
 * 
 * @param options The information specified by the user.
 * @param newickString The network in Extended Newick format.
 */
AnnotatedNetwork NetraxInstance::build_annotated_network_from_string(const NetraxOptions &options,
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
AnnotatedNetwork NetraxInstance::build_annotated_network_from_utree(const NetraxOptions &options,
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
AnnotatedNetwork NetraxInstance::build_random_annotated_network(const NetraxOptions &options) {
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
AnnotatedNetwork NetraxInstance::build_parsimony_annotated_network(const NetraxOptions &options) {
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
AnnotatedNetwork NetraxInstance::build_best_raxml_annotated_network(const NetraxOptions &options) {
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
    assert(!ann_network.branch_probs.empty());
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
    double old_score = scoreNetwork(ann_network);
    netrax::computeLoglikelihood(ann_network, 0, 1, true);
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after updating reticulation probs: " << new_score << "\n";
    assert(new_score <= old_score);
}

/**
 * Re-infers the likelihood model parameters of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeModel(AnnotatedNetwork &ann_network) {
    double old_score = scoreNetwork(ann_network);
    ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after model optimization: " << new_score << "\n";
    assert(new_score <= old_score);
}

/**
 * Re-infers the branch lengths of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeBranches(AnnotatedNetwork &ann_network) {
    double old_score = scoreNetwork(ann_network);
    ann_network.raxml_treeinfo->optimize_branches(ann_network.options.lh_epsilon, 1);
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after branch length optimization: " << new_score << "\n";
    assert(new_score <= old_score);
}

/**
 * Re-infers the topology of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeTopology(AnnotatedNetwork &ann_network, const std::vector<MoveType>& types) {
    double old_score = scoreNetwork(ann_network);
    greedyHillClimbingTopology(ann_network, types);
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC after topology optimization: " << new_score << "\n";
    assert(new_score <= old_score);
}

/**
 * Re-infers the topology of a given network.
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeTopology(AnnotatedNetwork &ann_network, MoveType& type) {
    double old_score = scoreNetwork(ann_network);
    greedyHillClimbingTopology(ann_network, type);
    double new_score = scoreNetwork(ann_network);
    //std::cout << "BIC after topology optimization: " << new_score << "\n";
    assert(new_score <= old_score);
}

/**
 * Re-infers everything (brach lengths, reticulation probabilities, likelihood model parameters, topology).
 * 
 * @param ann_network The network.
 */
void NetraxInstance::optimizeEverything(AnnotatedNetwork &ann_network) {
    std::vector<MoveType> typesBySpeed = {MoveType::DeltaMinusMove, MoveType::ArcRemovalMove, MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove, MoveType::DeltaPlusMove, MoveType::ArcInsertionMove};

    double score_epsilon = ann_network.options.lh_epsilon;
    unsigned int max_seconds = ann_network.options.timeout;
    auto start_time = std::chrono::high_resolution_clock::now();

    //std::cout << "Initial network loglikelihood: " << computeLoglikelihood(ann_network) << "\n";
    //double initial_score = scoreNetwork(ann_network);
    //std::cout << "Initial network BIC score: " << initial_score << "\n";

    optimizeBranches(ann_network);
    updateReticulationProbs(ann_network);
    optimizeModel(ann_network);
    double new_score = scoreNetwork(ann_network);
    std::cout << "Initial optimized network loglikelihood: " << computeLoglikelihood(ann_network) << "\n";
    std::cout << "Initial optimized network BIC score: " << new_score << "\n";
    //assert(new_score <= initial_score);

    //double_check_likelihood(ann_network);

    double old_score;

    unsigned int type_idx = 0;
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

        old_score = new_score;
        optimizeTopology(ann_network, typesBySpeed[type_idx]);
        new_score = scoreNetwork(ann_network);
        if (old_score - new_score > score_epsilon) { // score got better
            //std::cout << "BIC after topology optimization: " << new_score << "\n";
            //std::cout << "Current number of reticulations: " << ann_network.network.num_reticulations() << "\n";
            //std::cout << "network (BIC = " << new_score << ", logl = " << computeLoglikelihood(ann_network) << ") before brlen opt:\n" << toExtendedNewick(ann_network.network) << "\n";
            optimizeBranches(ann_network);
            //new_score = scoreNetwork(ann_network);
            //std::cout << "network (BIC = " << new_score << ", logl = " << computeLoglikelihood(ann_network) << ") after brlen opt:\n" << toExtendedNewick(ann_network.network) << "\n\n";
            updateReticulationProbs(ann_network);
            optimizeModel(ann_network);
            new_score = scoreNetwork(ann_network);
        }
        assert(new_score <= old_score);

        if (old_score - new_score > score_epsilon) { // score got better
            type_idx = 0; // go back to fastest move type
        } else { // try next-slower move type
            type_idx++;
        }

        if (max_seconds != 0) {
            auto act_time = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>( act_time - start_time ).count() >= max_seconds) {
                break;
            }
        }
    } while (type_idx < typesBySpeed.size());

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
    // If we have unlinked branch lenghts/probs, replace the entries in the network by their average
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED
            && ann_network.fake_treeinfo->partition_count > 1) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            double lenSum = 0.0;
            double probSum = 0.0;
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                lenSum += ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
                probSum += ann_network.branch_probs[p][pmatrix_index];
            }
            ann_network.network.edges[i].length = lenSum
                    / ann_network.fake_treeinfo->partition_count;
            ann_network.network.edges[i].prob = probSum
                    / ann_network.fake_treeinfo->partition_count;
        }
    }

    outfile << netrax::toExtendedNewick(ann_network.network) << "\n";
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
        std::cout << exportDebugInfo(ann_network.network) << "\n";
        std::cout << "reread_network:\n" << toExtendedNewick(ann_network2.network) << "\n";
        std::cout << exportDebugInfo(ann_network2.network) << "\n";
    }

    assert(similar_logl);
}

}
