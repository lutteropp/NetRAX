#include "AnnotatedNetwork.hpp"

#include <stddef.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

#include <raxml-ng/TreeInfo.hpp>
#include <raxml-ng/main.hpp>
#include "Network.hpp"
#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"
#include "../io/NetworkIO.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../NetraxOptions.hpp"
#include "../RaxmlWrapper.hpp"
#include "../moves/Move.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/ReticulationOptimization.hpp"
#include "../optimization/ModelOptimization.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../utils.hpp"

#include "NodeDisplayedTreeData.hpp"

namespace netrax {

AnnotatedNetwork::AnnotatedNetwork(NetraxOptions& options, const RaxmlInstance& instance) : options{options}, instance{instance} {}

AnnotatedNetwork::~AnnotatedNetwork() {
    if (fake_treeinfo) {
        destroy_network_treeinfo(fake_treeinfo);
    }
for (size_t p = 0; p < tmp_clv_1.size(); ++p) {
        pll_aligned_free(tmp_clv_1[p]);
        pll_aligned_free(tmp_clv_2[p]);
        pll_aligned_free(tmp_clv_3[p]);
    }
}

void destroy_network_treeinfo(pllmod_treeinfo_t *treeinfo) {
    if (!treeinfo)
        return;
    if (treeinfo->likelihood_computation_params != treeinfo) {
        free(treeinfo->likelihood_computation_params);
    }
    pllmod_treeinfo_destroy(treeinfo);
}

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
    ann_network.rng = rng;

    ann_network.fake_treeinfo = createNetworkPllTreeinfo(ann_network);

    ann_network.travbuffer = netrax::reversed_topological_sort(ann_network.network);

    allocateBranchProbsArray(ann_network);
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        double firstParentProb = ann_network.network.edges_by_index[ann_network.network.reticulation_nodes[i]->getReticulationData()->getLinkToFirstParent()->edge_pmatrix_index]->prob;
        ann_network.reticulation_probs[i] = firstParentProb;
    }

    ann_network.fake_treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    pllmod_treeinfo_update_prob_matrices(ann_network.fake_treeinfo, 1);
    for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[i]) {
            continue;
        }

        for (size_t j = 0; j < ann_network.network.num_tips(); ++j) { // tip nodes always have valid clv
            ann_network.fake_treeinfo->clv_valid[i][j] = 1;
        }
        for (size_t j = ann_network.network.num_tips(); j < ann_network.network.nodes.size(); ++j) { // nodes.size here, to also take care of currently unused nodes
            ann_network.fake_treeinfo->clv_valid[i][j] = 0;
        }
    }

    ann_network.partition_clv_ranges.resize(ann_network.fake_treeinfo->partition_count);
    ann_network.partition_scale_buffer_ranges.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        if (ann_network.fake_treeinfo->partitions[p]) { // ignore remote partitions
            ann_network.partition_clv_ranges[p] = get_clv_range(ann_network.fake_treeinfo->partitions[p]);
            ann_network.partition_scale_buffer_ranges[p] = get_scale_buffer_range(ann_network.fake_treeinfo->partitions[p]);
        }
    }

    ann_network.pernode_displayed_tree_data.resize(ann_network.network.nodes.size()); // including all nodes that will ever be there

    for (size_t i = 0; i < ann_network.network.num_tips(); ++i) {
        std::vector<double*> tip_clv(ann_network.fake_treeinfo->partition_count, nullptr);
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            tip_clv[p] = ann_network.fake_treeinfo->partitions[p]->clv[i];
        }
        ann_network.pernode_displayed_tree_data[i].displayed_trees.emplace_back(DisplayedTreeData(ann_network.fake_treeinfo, ann_network.partition_clv_ranges, ann_network.partition_scale_buffer_ranges, tip_clv, ann_network.options.max_reticulations));
        ann_network.pernode_displayed_tree_data[i].num_active_displayed_trees = 1;

        assert(ann_network.pernode_displayed_tree_data[i].displayed_trees.size() == 1);
        assert(ann_network.pernode_displayed_tree_data[i].displayed_trees[0].isTip);
    }

    ann_network.pseudo_clv_valid = std::vector<bool>(ann_network.network.nodes.size(), false);
    for (size_t i = 0; i < ann_network.network.num_tips(); ++i) {
        ann_network.pseudo_clv_valid[i] = true;
    }

    // we need at most 3 temporary clv vectors for pseudologlikelihood, allocate them now
    ann_network.tmp_clv_1.resize(ann_network.fake_treeinfo->partition_count);
    ann_network.tmp_clv_2.resize(ann_network.fake_treeinfo->partition_count);
    ann_network.tmp_clv_3.resize(ann_network.fake_treeinfo->partition_count);

    // allocate the temporary clv vectors
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        ann_network.tmp_clv_1[p] = create_single_empty_clv(ann_network.partition_clv_ranges[p]);
        ann_network.tmp_clv_2[p] = create_single_empty_clv(ann_network.partition_clv_ranges[p]);
        ann_network.tmp_clv_3[p] = create_single_empty_clv(ann_network.partition_clv_ranges[p]);
    }
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
AnnotatedNetwork build_annotated_network(NetraxOptions &options, const RaxmlInstance& instance) {
    AnnotatedNetwork ann_network(options, instance);
    ann_network.network = std::move(netrax::readNetworkFromFile(options.start_network_file, options,
            options.max_reticulations));
    return ann_network;
}

/**
 * Creates the inital annotated network data structure to work on, reading the network from a given string and ignoring the input file specified in the options.
 * 
 * @param options The information specified by the user.
 * @param newickString The network in Extended Newick format.
 */
AnnotatedNetwork build_annotated_network_from_string(NetraxOptions &options, const RaxmlInstance& instance,
        const std::string &newickString) {
    AnnotatedNetwork ann_network(options, instance);
    ann_network.network = netrax::readNetworkFromString(newickString, options, options.max_reticulations);
    return ann_network;
}

/**
 * Creates the inital annotated network data structure to work on, reading the network from a given file.
 * 
 * @param options The information specified by the user.
 * @param newickPath The path to the network in Extended Newick format.
 */
AnnotatedNetwork build_annotated_network_from_file(NetraxOptions &options, const RaxmlInstance& instance,
        const std::string &networkPath) {
    AnnotatedNetwork ann_network(options, instance);
    ann_network.network = std::move(netrax::readNetworkFromFile(networkPath, options,
            options.max_reticulations));
    return ann_network;
}

/**
 * Converts a pll_utree_t into an annotated network.
 * 
 * @param options The options specified by the user.
 * @param utree The pll_utree_t to be converted into an annotated network.
 */
AnnotatedNetwork build_annotated_network_from_utree(NetraxOptions &options, const RaxmlInstance& instance,
        const pll_utree_t &utree) {
    AnnotatedNetwork ann_network(options, instance);
    ann_network.network = netrax::convertUtreeToNetwork(utree, options, options.max_reticulations);
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
        std::vector<Move> candidates = possibleMoves(ann_network, MoveType::ArcInsertionMove);
        assert(!candidates.empty());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, candidates.size() - 1);
        Move move = candidates[dist(rng)];
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
AnnotatedNetwork build_random_annotated_network(NetraxOptions &options, const RaxmlInstance& instance, double seed) {
    Tree tree = generateRandomTree(instance, seed);
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, instance, tree.pll_utree());
    return ann_network;
}

/**
 * Creates an annotated network by building a maximum parsimony tree..
 * 
 * @param options The options specified by the user.
 */
AnnotatedNetwork build_parsimony_annotated_network(NetraxOptions &options, const RaxmlInstance& instance, double seed) {
    Tree tree = generateParsimonyTree(instance, seed);
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, instance, tree.pll_utree());
    return ann_network;
}

/**
 * Creates an annotated network by building a maximum likelihood tree..
 * 
 * @param options The options specified by the user.
 */
AnnotatedNetwork build_best_raxml_annotated_network(NetraxOptions &options, const RaxmlInstance& instance) {
    Tree tree = bestRaxmlTree(instance);
    AnnotatedNetwork ann_network = build_annotated_network_from_utree(options, instance, tree.pll_utree());
    return ann_network;
}

AnnotatedNetwork::AnnotatedNetwork(const AnnotatedNetwork& orig_network) : options{orig_network.options}, instance{orig_network.instance} {
    network = orig_network.network;
    init_annotated_network(*this);
}

bool clvValidCheck(AnnotatedNetwork& ann_network, size_t virtual_root_clv_index) {
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool all_clvs_valid = true;
    for (size_t p = 0; p < fake_treeinfo.partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        all_clvs_valid &= fake_treeinfo.clv_valid[p][virtual_root_clv_index];
    }
    return all_clvs_valid;
}

bool reuseOldDisplayedTreesCheck(AnnotatedNetwork& ann_network, int incremental, size_t virtual_root_clv_index) {
    if (!incremental) {
        return false;
    }
    return clvValidCheck(ann_network, virtual_root_clv_index);
}

bool hasBadReticulation(AnnotatedNetwork& ann_network) {
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        if ((1.0 - ann_network.reticulation_probs[i] < 0.001) || (ann_network.reticulation_probs[i] < 0.001)) {
            return true;
        }
    }
    return false;
}

std::vector<Node*> getBadReticulations(AnnotatedNetwork& ann_network) {
    std::vector<Node*> res;
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        if ((1.0 - ann_network.reticulation_probs[i] < 0.001) || (ann_network.reticulation_probs[i] < 0.001)) {
            res.emplace_back(ann_network.network.reticulation_nodes[i]);
        }
    }
    return res;
}

bool assertBranchLengths(AnnotatedNetwork& ann_network) {
    if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
                assert(ann_network.fake_treeinfo->branch_lengths[p][i] >= ann_network.options.brlen_min);
                assert(ann_network.fake_treeinfo->branch_lengths[p][i] <= ann_network.options.brlen_max);
            }
        }
    } else {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            assert(ann_network.fake_treeinfo->linked_branch_lengths[i] >= ann_network.options.brlen_min);
            assert(ann_network.fake_treeinfo->linked_branch_lengths[i] <= ann_network.options.brlen_max);
        }
    }
    return true;
}

bool assertConsecutiveIndices(AnnotatedNetwork& ann_network) {
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        assert(ann_network.network.nodes_by_index[i]);
    }
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        assert(ann_network.network.edges_by_index[i]);
    }
    return true;
}

bool checkSanity(AnnotatedNetwork& ann_network) {
    bool good = true;
    good &= checkSanity(ann_network.network);
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++i) {
            DisplayedTreeData& dtd = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[i];
            assert(!dtd.treeLoglData.reticulationChoices.configs.empty());
        }
    }
    return good;
}

}