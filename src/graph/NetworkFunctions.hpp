/*
 * NetworkFunctions.hpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>
#include <unordered_map>

extern "C"
{
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "Node.hpp"

namespace netrax {
class Network;
} /* namespace netrax */

namespace netrax {

std::vector<std::vector<size_t> > getDtBranchToNetworkBranchMapping(const pll_utree_t& utree, Network& network, size_t tree_idx);

pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index);
std::vector<double> collectBranchLengths(const Network& network);
void applyBranchLengths(Network& network, const std::vector<double>& branchLengths);
void setReticulationParents(Network& network, size_t treeIdx);
/*
 * Find possible placements for the root node in a semi-rooted network.
 */
std::vector<Node*> getPossibleRootNodes(Network& network);

void fill_dead_nodes_recursive(const Node* myParent, const Node* node, std::vector<bool>& dead_nodes);
}
