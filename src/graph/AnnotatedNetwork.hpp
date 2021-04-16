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
    std::vector<ClvRangeInfo> clvInfo;
    std::vector<ScaleBufferRangeInfo> scaleBufferInfo;

    void add_displayed_tree(pllmod_treeinfo_t* treeinfo, const std::vector<ClvRangeInfo>& clvInfo, const std::vector<ScaleBufferRangeInfo>& scaleBufferInfo, size_t maxReticulations) {
        this->clvInfo = clvInfo;
        this->scaleBufferInfo = scaleBufferInfo;
        num_active_displayed_trees++;
        if (num_active_displayed_trees > displayed_trees.size()) {
            assert(num_active_displayed_trees == displayed_trees.size() + 1);
            displayed_trees.emplace_back(DisplayedTreeData(treeinfo, clvInfo, scaleBufferInfo, maxReticulations));
        } else { // zero out the clv vector and scale buffer
            for (size_t p = 0; p < treeinfo->partition_count; ++p) {
                /* skip remote partitions */
                if (!treeinfo->partitions[p]) {
                    continue;
                }

                assert(displayed_trees[num_active_displayed_trees-1].clv_vector[p]);
                memset(displayed_trees[num_active_displayed_trees-1].clv_vector[p], 0, clvInfo[p].inner_clv_num_entries * sizeof(double));
                if (displayed_trees[num_active_displayed_trees-1].scale_buffer[p]) {
                    memset(displayed_trees[num_active_displayed_trees-1].scale_buffer[p], 0, scaleBufferInfo[p].scaler_size * sizeof(unsigned int));
                }
            }
        }
    }

    bool operator==(const NodeDisplayedTreeData& rhs) const
    {
        if (num_active_displayed_trees != rhs.num_active_displayed_trees) {
            return false;
        }
        for (size_t i = 0; i < num_active_displayed_trees; ++i) {
            for (size_t p = 0; p < clvInfo.size(); ++p) {
                if (!clv_single_entries_equal(clvInfo[p], displayed_trees[i].clv_vector[p], rhs.displayed_trees[i].clv_vector[p])) {
                    return false;
                }
            }
            for (size_t p = 0; p < scaleBufferInfo.size(); ++p) {
                if (!scale_buffer_single_entries_equal(scaleBufferInfo[p], displayed_trees[i].scale_buffer[p], rhs.displayed_trees[i].scale_buffer[p])) {
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

    NodeDisplayedTreeData(std::vector<ClvRangeInfo>& clvInfo, std::vector<ScaleBufferRangeInfo>& scaleBufferInfo) : clvInfo{clvInfo}, scaleBufferInfo{scaleBufferInfo} {
        num_active_displayed_trees = 0;
        displayed_trees = std::vector<DisplayedTreeData>();
    }

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
    const NetraxOptions& options;
    Network network; // The network topology itself
    
    size_t total_num_model_parameters = 0;
    size_t total_num_sites = 0;
    pllmod_treeinfo_t *fake_treeinfo = nullptr;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    std::vector<double> partition_contributions;

    std::vector<NodeDisplayedTreeData> pernode_displayed_tree_data;

    std::vector<Node*> travbuffer;
    std::mt19937 rng;

    Statistics stats;

    double cached_logl = -std::numeric_limits<int>::infinity();
    bool cached_logl_valid = false;

    AnnotatedNetwork(AnnotatedNetwork&&) = default;
    AnnotatedNetwork(const NetraxOptions& options) : options{options} {};

    ~AnnotatedNetwork() {
        if (fake_treeinfo) {
            destroy_network_treeinfo(fake_treeinfo);
        }
    }

    AnnotatedNetwork(const AnnotatedNetwork& orig_network);
};

AnnotatedNetwork build_annotated_network(NetraxOptions &options);
AnnotatedNetwork build_annotated_network_from_string(NetraxOptions &options,
        const std::string &newickString);
AnnotatedNetwork build_annotated_network_from_file(NetraxOptions &options,
        const std::string &networkPath);
AnnotatedNetwork build_annotated_network_from_utree(NetraxOptions &options,
        const pll_utree_t &utree);
AnnotatedNetwork build_random_annotated_network(NetraxOptions &options, double seed);
AnnotatedNetwork build_parsimony_annotated_network(NetraxOptions &options, double seed);
AnnotatedNetwork build_best_raxml_annotated_network(NetraxOptions &options);
void add_extra_reticulations(AnnotatedNetwork &ann_network, unsigned int targetCount);
void init_annotated_network(AnnotatedNetwork &ann_network, std::mt19937& rng);
void init_annotated_network(AnnotatedNetwork &ann_network);

}
