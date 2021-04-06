/*
 * BranchLengthOptimization.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../graph/Network.hpp"
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

#include <unordered_set>

extern "C" {
#include <libpll/pll_tree.h>
}

namespace netrax {

void add_neighbors_in_radius(AnnotatedNetwork& ann_network, std::unordered_set<size_t>& candidates, int radius);
void add_neighbors_in_radius(AnnotatedNetwork& ann_network, std::unordered_set<size_t>& candidates, size_t pmatrix_index, int radius);

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius, bool most_likely_tree_only = false);
double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius,
        std::unordered_set<size_t> candidates, bool most_likely_tree_only = false);

std::vector<std::vector<TreeLoglData> > extractOldTrees(AnnotatedNetwork& ann_network, Node* virtual_root);

void optimizeBranches(AnnotatedNetwork &ann_network, bool silent = true, bool most_likely_tree_only = false);

}
