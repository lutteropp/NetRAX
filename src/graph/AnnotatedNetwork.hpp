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

namespace netrax {

struct NetworkState;
struct NetraxOptions;
struct ScaleBufferRangeInfo;
struct ClvRangeInfo;
enum class MoveType;
struct DisplayedTreeData;
struct NodeDisplayedTreeData;

struct Statistics {
    std::unordered_map<MoveType, size_t> moves_taken;
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
    AnnotatedNetwork(NetraxOptions& options, const RaxmlInstance& instance);
    AnnotatedNetwork(const AnnotatedNetwork& orig_network);
    ~AnnotatedNetwork();
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

bool clvValidCheck(AnnotatedNetwork& ann_network, size_t virtual_root_clv_index);
bool reuseOldDisplayedTreesCheck(AnnotatedNetwork& ann_network, int incremental, size_t virtual_root_clv_index);
bool hasBadReticulation(AnnotatedNetwork& ann_network);
std::vector<Node*> getBadReticulations(AnnotatedNetwork& ann_network);
bool assertBranchLengths(AnnotatedNetwork& ann_network);
bool assertConsecutiveIndices(AnnotatedNetwork& ann_network);
bool checkSanity(AnnotatedNetwork& ann_network);

}
