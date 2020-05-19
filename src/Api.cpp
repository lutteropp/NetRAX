/*
 * Api.cpp
 *
 *  Created on: Apr 24, 2020
 *      Author: sarah
 */

#include "Api.hpp"

#include <stddef.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

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
#include "graph/Moves.hpp"
#include "optimization/TopologyOptimization.hpp"

namespace netrax {

void init_annotated_network(AnnotatedNetwork &ann_network) {
    Network &network = ann_network.network;
    std::random_device dev;
    std::mt19937 rng(dev());
    ann_network.rng = rng;

    netrax::RaxmlWrapper wrapper(ann_network.options);
    ann_network.raxml_treeinfo = std::unique_ptr<TreeInfo>(wrapper.createRaxmlTreeinfo(ann_network));
    ann_network.travbuffer = netrax::reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = netrax::partitionNetworkIntoBlobs(network, ann_network.travbuffer);

    // init branch probs...
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) { // common branches
        ann_network.branch_probs = std::vector<std::vector<double> >(1,
                std::vector<double>(ann_network.network.edges.size() + 1, 1.0));
    } else { // each partition has extra branch properties
        ann_network.branch_probs = std::vector<std::vector<double> >(ann_network.fake_treeinfo->partition_count,
                std::vector<double>(ann_network.network.edges.size() + 1, 1.0));
    }
    for (size_t p = 0; p < ann_network.branch_probs.size(); ++p) {
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            Node *retNode = ann_network.network.reticulation_nodes[i];
            double firstParentProb = netrax::getReticulationFirstParentProb(network, retNode);
            double secondParentProb = netrax::getReticulationSecondParentProb(network, retNode);
            size_t firstParentEdgeIndex = netrax::getReticulationFirstParentPmatrixIndex(network, retNode);
            size_t secondParentEdgeIndex = netrax::getReticulationSecondParentPmatrixIndex(network, retNode);
            ann_network.branch_probs[p][firstParentEdgeIndex] = firstParentProb;
            ann_network.branch_probs[p][secondParentEdgeIndex] = secondParentProb;
        }
    }
}

AnnotatedNetwork build_annotated_network(const NetraxOptions &options) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = netrax::readNetworkFromFile(options.network_file, options.max_reticulations);
    init_annotated_network(ann_network);
    return ann_network;
}

AnnotatedNetwork build_annotated_network_from_string(const NetraxOptions &options, const std::string &newickString) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = netrax::readNetworkFromString(newickString, options.max_reticulations);
    init_annotated_network(ann_network);
    return ann_network;
}

AnnotatedNetwork build_annotated_network_from_utree(const NetraxOptions &options, const pll_utree_t &utree) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = netrax::convertUtreeToNetwork(utree, options.max_reticulations);
    init_annotated_network(ann_network);
    return ann_network;
}

void add_extra_reticulations(AnnotatedNetwork &ann_network, unsigned int targetCount) {
    if (targetCount > ann_network.options.max_reticulations) {
        throw std::runtime_error("Please increase maximum allowed number of reticulations");
    }
    Network &network = ann_network.network;
    std::mt19937 &rng = ann_network.rng;

    // TODO: This can be implemented more eficiently than by always re-gathering all candidates
    while (targetCount > network.num_reticulations()) {
        std::vector<ArcInsertionMove> candidates = possibleArcInsertionMoves(ann_network);
        assert(!candidates.empty());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, candidates.size() - 1);
        ArcInsertionMove move = candidates[dist(rng)];
        performMove(ann_network, move);
    }
}

AnnotatedNetwork build_random_annotated_network(const NetraxOptions &options, unsigned int start_reticulations) {
    RaxmlWrapper wrapper(options);
    Tree tree = wrapper.generateRandomTree();
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, tree.pll_utree());
    add_extra_reticulations(ann_network, start_reticulations);
    return ann_network;
}

AnnotatedNetwork build_parsimony_annotated_network(const NetraxOptions &options, unsigned int start_reticulations) {
    RaxmlWrapper wrapper(options);
    Tree tree = wrapper.generateParsimonyTree();
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, tree.pll_utree());
    add_extra_reticulations(ann_network, start_reticulations);
    return ann_network;
}

AnnotatedNetwork build_best_raxml_annotated_network(const NetraxOptions &options, unsigned int start_reticulations) {
    throw std::runtime_error("Not implemented yet");
}

double computeLoglikelihood(AnnotatedNetwork &ann_network) {
    return ann_network.raxml_treeinfo->loglh(true);
}
double updateReticulationProbs(AnnotatedNetwork &ann_network) {
    return netrax::computeLoglikelihood(ann_network, 0, 1, true);
}
double optimizeModel(AnnotatedNetwork &ann_network) {
    return ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
}
double optimizeBranches(AnnotatedNetwork &ann_network) {
    return ann_network.raxml_treeinfo->optimize_branches(ann_network.options.lh_epsilon, 1);
}
double optimizeTopology(AnnotatedNetwork &ann_network) {
    return searchBetterTopology(ann_network);
}

void writeNetwork(AnnotatedNetwork &ann_network, const std::string &filepath) {
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
            ann_network.network.edges[i].length = lenSum / ann_network.fake_treeinfo->partition_count;
            ann_network.network.edges[i].prob = probSum / ann_network.fake_treeinfo->partition_count;
        }
    }

    outfile << netrax::toExtendedNewick(ann_network.network) << "\n";
    outfile.close();
}

}
