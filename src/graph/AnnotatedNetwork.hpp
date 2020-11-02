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

namespace netrax {

struct Statistics {
    std::unordered_map<MoveType, size_t> moves_taken;
};

struct AnnotatedNetwork {
    Network network; // The network topology itself
    size_t total_num_model_parameters;
    size_t total_num_sites;
    std::unique_ptr<TreeInfo> raxml_treeinfo = nullptr;
    pllmod_treeinfo_t *fake_treeinfo = nullptr;
    NetraxOptions options;
    BlobInformation blobInfo; // mapping of edges to blobs, megablob roots, mapping of megablob roots to set of reticulation nodes within the megablob
    std::vector<std::vector<double> > branch_probs; // for each partition, the branch length probs
    double old_logl;
    std::vector<Node*> travbuffer;
    std::mt19937 rng;

    Statistics stats;
};

}
