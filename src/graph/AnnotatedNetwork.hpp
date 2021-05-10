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
#include <raxml-ng/main.hpp>
#include "Network.hpp"
#include "../NetraxOptions.hpp"
#include "../optimization/MoveType.hpp"
#include "../likelihood/DisplayedTreeData.hpp"

namespace netrax {

struct NetworkState;

struct Statistics {
    std::unordered_map<MoveType, size_t> moves_taken;
};

struct NodeDisplayedTreeData {
    std::vector<DisplayedTreeData> displayed_trees;
    size_t num_active_displayed_trees = 0;
    size_t partition_count = 0;
    std::vector<ClvRangeInfo> clvInfo;
    std::vector<ScaleBufferRangeInfo> scaleBufferInfo;

    bool operator==(const NodeDisplayedTreeData& rhs) const
    {
        if (num_active_displayed_trees != rhs.num_active_displayed_trees) {
            return false;
        }
        for (size_t i = 0; i < num_active_displayed_trees; ++i) {
            for (size_t j = 0; j < partition_count; ++j) {
                if (!clv_single_entries_equal(clvInfo[j], displayed_trees[i].clv_vector[j], rhs.displayed_trees[i].clv_vector[j])) {
                    return false;
                }
                if (!scale_buffer_single_entries_equal(scaleBufferInfo[j], displayed_trees[i].scale_buffer[j], rhs.displayed_trees[i].scale_buffer[j])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool operator!=(const NodeDisplayedTreeData& rhs) const
    {
        return !operator==(rhs);
    }

    NodeDisplayedTreeData(NodeDisplayedTreeData&& rhs)
      : displayed_trees{rhs.displayed_trees}, num_active_displayed_trees{rhs.num_active_displayed_trees}, partition_count{rhs.partition_count}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}
    {
        rhs.num_active_displayed_trees = 0;
        rhs.displayed_trees = std::vector<DisplayedTreeData>();
    }

    NodeDisplayedTreeData(const NodeDisplayedTreeData& rhs)
      : num_active_displayed_trees{rhs.num_active_displayed_trees}, partition_count{rhs.partition_count}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}
    {
        displayed_trees = rhs.displayed_trees;
    }

    NodeDisplayedTreeData() = default;

    NodeDisplayedTreeData& operator =(NodeDisplayedTreeData&& rhs)
    {
        if (this != &rhs)
        {
            displayed_trees = std::move(rhs.displayed_trees);
            num_active_displayed_trees = rhs.num_active_displayed_trees;
            partition_count = rhs.partition_count;
            rhs.num_active_displayed_trees = 0;
            clvInfo = rhs.clvInfo;
            scaleBufferInfo = rhs.scaleBufferInfo;
        }
        return *this;
    }

    NodeDisplayedTreeData& operator =(const NodeDisplayedTreeData& rhs)
    {
        if (this != &rhs)
        {
            for (size_t i = 0; i < std::min(displayed_trees.size(), rhs.displayed_trees.size()); ++i) {
                displayed_trees[i] = rhs.displayed_trees[i];
            }
            for (size_t i = std::min(displayed_trees.size(), rhs.displayed_trees.size()); i < rhs.displayed_trees.size(); ++i) {
                displayed_trees.emplace_back(rhs.displayed_trees[i]);
            }
            num_active_displayed_trees = rhs.num_active_displayed_trees;
            partition_count = rhs.partition_count;
            clvInfo = rhs.clvInfo;
            scaleBufferInfo = rhs.scaleBufferInfo;
        }
        return *this;
    }
};

void destroy_network_treeinfo(pllmod_treeinfo_t *treeinfo);

struct AnnotatedNetwork {
    NetraxOptions& options;
    const RaxmlInstance& instance;
    Network network; // The network topology itself
    
    size_t total_num_model_parameters = 0;
    size_t total_num_sites = 0;
    pllmod_treeinfo_t *fake_treeinfo = nullptr;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    std::vector<double> partition_contributions;

    std::vector<ClvRangeInfo> partition_clv_ranges;
    std::vector<ScaleBufferRangeInfo> partition_scale_buffer_ranges;

    std::vector<NodeDisplayedTreeData> pernode_displayed_tree_data;

    std::vector<Node*> travbuffer;
    std::mt19937 rng;

    std::vector<bool> pseudo_clv_valid;
    std::vector<double*> tmp_clv_1;
    std::vector<double*> tmp_clv_2;
    std::vector<double*> tmp_clv_3;

    size_t last_accepted_move_edge_orig_idx = std::numeric_limits<size_t>::infinity();

    Statistics stats;

    double cached_logl = -std::numeric_limits<int>::infinity();
    bool cached_logl_valid = false;

    AnnotatedNetwork(AnnotatedNetwork&&) = default;
    AnnotatedNetwork(NetraxOptions& options, const RaxmlInstance& instance) : options{options}, instance{instance} {}

    ~AnnotatedNetwork() {
        if (fake_treeinfo) {
            destroy_network_treeinfo(fake_treeinfo);
        }
	for (size_t p = 0; p < tmp_clv_1.size(); ++p) {
		pll_aligned_free(tmp_clv_1[p]);
		pll_aligned_free(tmp_clv_2[p]);
		pll_aligned_free(tmp_clv_3[p]);
    	}
    }

    AnnotatedNetwork(const AnnotatedNetwork& orig_network);
};

AnnotatedNetwork build_annotated_network(NetraxOptions &options, const RaxmlInstance& instance);
AnnotatedNetwork build_annotated_network_from_string(NetraxOptions &options, const RaxmlInstance& instance,
        const std::string &newickString);
AnnotatedNetwork build_annotated_network_from_file(NetraxOptions &options, const RaxmlInstance& instance,
        const std::string &networkPath);
AnnotatedNetwork build_annotated_network_from_utree(NetraxOptions &options, const RaxmlInstance& instance,
        const pll_utree_t &utree);
AnnotatedNetwork build_random_annotated_network(NetraxOptions &options, const RaxmlInstance& instance, double seed);
AnnotatedNetwork build_parsimony_annotated_network(NetraxOptions &options, const RaxmlInstance& instance,  double seed);
AnnotatedNetwork build_best_raxml_annotated_network(NetraxOptions &options, const RaxmlInstance& instance);
void add_extra_reticulations(AnnotatedNetwork &ann_network, unsigned int targetCount);
void init_annotated_network(AnnotatedNetwork &ann_network, std::mt19937& rng);
void init_annotated_network(AnnotatedNetwork &ann_network);

bool reuseOldDisplayedTreesCheck(AnnotatedNetwork& ann_network, int incremental);

}
