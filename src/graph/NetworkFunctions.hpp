/*
 * NetworkFunctions.hpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

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

pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index);
std::vector<double> collectBranchLengths(const Network& network);
void applyBranchLengths(Network& network, const std::vector<double>& branchLengths);
void setReticulationParents(Network& network, size_t treeIdx);
/*
 * Find possible placements for the root node in a semi-rooted network.
 */
std::vector<const Node*> getPossibleRootNodes(const Network& network);
}
