#include "AnnotatedNetwork.hpp"

#include <stddef.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

#include <raxml-ng/TreeInfo.hpp>
#include <raxml-ng/main.hpp>
#include "BiconnectedComponents.hpp"
#include "Network.hpp"
#include "NetworkFunctions.hpp"
#include "NetworkTopology.hpp"
#include "../io/NetworkIO.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../NetraxOptions.hpp"
#include "../RaxmlWrapper.hpp"
#include "../optimization/Moves.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/ReticulationOptimization.hpp"
#include "../optimization/ModelOptimization.hpp"
#include "../optimization/TopologyOptimization.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../utils.hpp"

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
void init_annotated_network(AnnotatedNetwork &ann_network, std::mt19937& rng) {
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
        for (size_t j = 0; j < ann_network.network.num_tips(); ++j) { // tip nodes always have valid clv
            ann_network.fake_treeinfo->clv_valid[i][j] = 1;
        }
        for (size_t j = ann_network.network.num_tips(); j < ann_network.network.nodes.size(); ++j) { // nodes.size here, to also take care of currently unused nodes
            ann_network.fake_treeinfo->clv_valid[i][j] = 0;
        }
    }

    ann_network.displayed_trees.resize(ann_network.fake_treeinfo->partition_count);
    size_t n_trees = (1 << ann_network.network.num_reticulations());
    for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
        assert(!pll_repeats_enabled(ann_network.fake_treeinfo->partitions[i]));
        ann_network.displayed_trees[i].resize(n_trees);
        for (size_t j = 0; j < n_trees; ++j) {
            ann_network.displayed_trees[i][j].tree_clv_vectors = clone_clv_vector(ann_network.fake_treeinfo->partitions[i], ann_network.fake_treeinfo->partitions[i]->clv);
            ann_network.displayed_trees[i][j].tree_scale_buffers = clone_scale_buffer(ann_network.fake_treeinfo->partitions[i], ann_network.fake_treeinfo->partitions[i]->scale_buffer);
        }
    }

    ann_network.pernode_displayed_tree_data.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        ann_network.pernode_displayed_tree_data[p].resize(ann_network.network.nodes.size()); // including all nodes that will ever be there
        for (size_t i = 0; i < ann_network.network.num_tips(); ++i) {
            ann_network.pernode_displayed_tree_data[p][i].emplace_back(DisplayedTreeClvData(ann_network.fake_treeinfo->partitions[p]->clv[i], ann_network.options.max_reticulations));
        }
    }

    netrax::computeLoglikelihood(ann_network, false, true);
}

void init_annotated_network(AnnotatedNetwork &ann_network) {
    std::random_device dev;
    std::mt19937 rng(dev());
    init_annotated_network(ann_network, rng);
}

/**
 * Creates the inital annotated network data structure to work on, reading the network from the file specified in the options.
 * 
 * @param options The information specified by the user.
 */
AnnotatedNetwork build_annotated_network(NetraxOptions &options) {
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
AnnotatedNetwork build_annotated_network_from_string(NetraxOptions &options,
        const std::string &newickString) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = netrax::readNetworkFromString(newickString, options.max_reticulations);
    return ann_network;
}

/**
 * Creates the inital annotated network data structure to work on, reading the network from a given file.
 * 
 * @param options The information specified by the user.
 * @param newickPath The path to the network in Extended Newick format.
 */
AnnotatedNetwork build_annotated_network_from_file(NetraxOptions &options,
        const std::string &networkPath) {
    AnnotatedNetwork ann_network;
    ann_network.options = options;
    ann_network.network = std::move(netrax::readNetworkFromFile(networkPath,
            options.max_reticulations));
    return ann_network;
}

/**
 * Converts a pll_utree_t into an annotated network.
 * 
 * @param options The options specified by the user.
 * @param utree The pll_utree_t to be converted into an annotated network.
 */
AnnotatedNetwork build_annotated_network_from_utree(NetraxOptions &options,
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
void add_extra_reticulations(AnnotatedNetwork &ann_network, unsigned int targetCount) {
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
AnnotatedNetwork build_random_annotated_network(NetraxOptions &options) {
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
AnnotatedNetwork build_parsimony_annotated_network(NetraxOptions &options) {
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
AnnotatedNetwork build_best_raxml_annotated_network(NetraxOptions &options) {
    RaxmlWrapper wrapper(options);
    Tree tree = wrapper.bestRaxmlTree();
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, tree.pll_utree());
    return ann_network;
}

}