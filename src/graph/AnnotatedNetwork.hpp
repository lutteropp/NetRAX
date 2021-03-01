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

struct AnnotatedNetwork {
    NetraxOptions options;
    Network network; // The network topology itself
    
    size_t total_num_model_parameters = 0;
    size_t total_num_sites = 0;
    std::unique_ptr<TreeInfo> raxml_treeinfo = nullptr;
    pllmod_treeinfo_t *fake_treeinfo = nullptr;
    BlobInformation blobInfo; // mapping of edges to blobs, megablob roots, mapping of megablob roots to set of reticulation nodes within the megablob
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    std::vector<std::vector<DisplayedTreeData> > displayed_trees;

    std::vector<std::vector<DisplayedTreeClvData> > pernode_displayed_tree_data;

    std::vector<Node*> travbuffer;
    std::mt19937 rng;

    Statistics stats;

    AnnotatedNetwork(AnnotatedNetwork&&) = default;
    AnnotatedNetwork() = default;
    ~AnnotatedNetwork() {
        for (size_t i = 0; i < displayed_trees.size(); ++i) {
                for (size_t j = 0; j < displayed_trees[i].size(); ++j) {
                        delete_cloned_clv_vector(fake_treeinfo->partitions[i], displayed_trees[i][j].tree_clv_vectors);
                        delete_cloned_scale_buffer(fake_treeinfo->partitions[i], displayed_trees[i][j].tree_scale_buffers);
                }
        }
    }
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
