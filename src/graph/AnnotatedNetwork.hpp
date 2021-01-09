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

    std::vector<std::vector<DisplayedTreeData> > old_displayed_trees;
    std::vector<Node*> travbuffer;
    std::mt19937 rng;

    Statistics stats;
};

}
