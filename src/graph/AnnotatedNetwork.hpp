/*
 * AnnotatedNetwork.hpp
 *
 *  Created on: Feb 25, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <vector>
#include <memory>
#include <random>
#include <unordered_map>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}
#include <raxml-ng/TreeInfo.hpp>
#include "Network.hpp"
#include "NetworkFunctions.hpp"
#include "BiconnectedComponents.hpp"
#include "../NetraxOptions.hpp"
#include "../optimization/MoveType.hpp"
#include "../likelihood/DisplayedTreeData.hpp"

namespace netrax {

struct NetworkState;

struct Statistics {
    std::unordered_map<MoveType, size_t> moves_taken;
};

struct NodeDisplayedTreeData {
    std::vector<DisplayedTreeClvData> displayed_trees;
    size_t num_active_displayed_trees = 0;

    void add_displayed_tree(ClvRangeInfo clvInfo, ScaleBufferRangeInfo scaleBufferInfo, size_t maxReticulations) {
        num_active_displayed_trees++;
        if (num_active_displayed_trees > displayed_trees.size()) {
            assert(num_active_displayed_trees == displayed_trees.size() + 1);
            displayed_trees.emplace_back(DisplayedTreeClvData(clvInfo, scaleBufferInfo, maxReticulations));
        } else { // zero out the clv vector and scale buffer
            assert(displayed_trees[num_active_displayed_trees-1].clv_vector);
            memset(displayed_trees[num_active_displayed_trees-1].clv_vector, 0, clvInfo.inner_clv_num_entries * sizeof(double));
            if (displayed_trees[num_active_displayed_trees-1].scale_buffer) {
                memset(displayed_trees[num_active_displayed_trees-1].scale_buffer, 0, scaleBufferInfo.scaler_size * sizeof(unsigned int));
            }
        }
    }
};

struct AnnotatedNetwork {
    NetraxOptions options;
    Network network; // The network topology itself
    
    size_t total_num_model_parameters = 0;
    size_t total_num_sites = 0;
    std::unique_ptr<TreeInfo> raxml_treeinfo = nullptr;
    pllmod_treeinfo_t *fake_treeinfo = nullptr;
    BlobInformation blobInfo; // mapping of edges to blobs, megablob roots, mapping of megablob roots to set of reticulation nodes within the megablob
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    std::vector<std::vector<NodeDisplayedTreeData> > pernode_displayed_tree_data;

    std::vector<Node*> travbuffer;
    std::mt19937 rng;

    Statistics stats;

    AnnotatedNetwork(AnnotatedNetwork&&) = default;
    AnnotatedNetwork() = default;
};

AnnotatedNetwork build_annotated_network(NetraxOptions &options);
AnnotatedNetwork build_annotated_network_from_string(NetraxOptions &options,
        const std::string &newickString);
AnnotatedNetwork build_annotated_network_from_file(NetraxOptions &options,
        const std::string &networkPath);
AnnotatedNetwork build_annotated_network_from_utree(NetraxOptions &options,
        const pll_utree_t &utree);
AnnotatedNetwork build_random_annotated_network(NetraxOptions &options);
AnnotatedNetwork build_parsimony_annotated_network(NetraxOptions &options);
AnnotatedNetwork build_best_raxml_annotated_network(NetraxOptions &options);
void add_extra_reticulations(AnnotatedNetwork &ann_network, unsigned int targetCount);
void init_annotated_network(AnnotatedNetwork &ann_network, std::mt19937& rng);
void init_annotated_network(AnnotatedNetwork &ann_network);

}
