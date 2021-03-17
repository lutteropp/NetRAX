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
    ClvRangeInfo clvInfo;
    ScaleBufferRangeInfo scaleBufferInfo;

    void add_displayed_tree(ClvRangeInfo clvInfo, ScaleBufferRangeInfo scaleBufferInfo, size_t maxReticulations) {
        this->clvInfo = clvInfo;
        this->scaleBufferInfo = scaleBufferInfo;
        num_active_displayed_trees++;
        if (num_active_displayed_trees > displayed_trees.size()) {
            assert(num_active_displayed_trees == displayed_trees.size() + 1);
            displayed_trees.emplace_back(DisplayedTreeData(clvInfo, scaleBufferInfo, maxReticulations));
        } else { // zero out the clv vector and scale buffer
            assert(displayed_trees[num_active_displayed_trees-1].clv_vector);
            memset(displayed_trees[num_active_displayed_trees-1].clv_vector, 0, clvInfo.inner_clv_num_entries * sizeof(double));
            if (displayed_trees[num_active_displayed_trees-1].scale_buffer) {
                memset(displayed_trees[num_active_displayed_trees-1].scale_buffer, 0, scaleBufferInfo.scaler_size * sizeof(unsigned int));
            }
        }
    }

    bool operator==(const NodeDisplayedTreeData& rhs) const
    {
        if (num_active_displayed_trees != rhs.num_active_displayed_trees) {
            return false;
        }
        for (size_t i = 0; i < num_active_displayed_trees; ++i) {
            if (!clv_single_entries_equal(clvInfo, displayed_trees[i].clv_vector, rhs.displayed_trees[i].clv_vector)) {
                return false;
            }
            if (!scale_buffer_single_entries_equal(scaleBufferInfo, displayed_trees[i].scale_buffer, rhs.displayed_trees[i].scale_buffer)) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const NodeDisplayedTreeData& rhs) const
    {
        return !operator==(rhs);
    }

    NodeDisplayedTreeData(NodeDisplayedTreeData&& rhs)
      : displayed_trees{rhs.displayed_trees}, num_active_displayed_trees{rhs.num_active_displayed_trees}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}
    {
        rhs.num_active_displayed_trees = 0;
        rhs.displayed_trees = std::vector<DisplayedTreeData>();
    }

    NodeDisplayedTreeData(const NodeDisplayedTreeData& rhs)
      : num_active_displayed_trees{rhs.num_active_displayed_trees}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}
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
            displayed_trees = rhs.displayed_trees;
            num_active_displayed_trees = rhs.num_active_displayed_trees;
            clvInfo = rhs.clvInfo;
            scaleBufferInfo = rhs.scaleBufferInfo;
        }
        return *this;
    }
};

void destroy_network_treeinfo(pllmod_treeinfo_t *treeinfo);

struct AnnotatedNetwork {
    NetraxOptions options;
    Network network; // The network topology itself
    
    size_t total_num_model_parameters = 0;
    size_t total_num_sites = 0;
    pllmod_treeinfo_t *fake_treeinfo = nullptr;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    std::vector<double> partition_contributions;

    std::vector<std::vector<NodeDisplayedTreeData> > pernode_displayed_tree_data;

    std::vector<Node*> travbuffer;
    std::mt19937 rng;

    Statistics stats;

    double cached_logl = -std::numeric_limits<int>::infinity();
    bool cached_logl_valid = false;

    AnnotatedNetwork(AnnotatedNetwork&&) = default;
    AnnotatedNetwork() = default;

    ~AnnotatedNetwork() {
        if (fake_treeinfo) {
            destroy_network_treeinfo(fake_treeinfo);
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
