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
#include <string>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}
#include "../likelihood/DisplayedTreeData.hpp"

namespace netrax {

struct Node;
class Network;
class AnnotatedNetwork;
struct BlobInformation;

std::vector<std::vector<size_t> > getDtBranchToNetworkBranchMapping(const pll_utree_t &utree,
        Network &network, size_t tree_idx);

double displayed_tree_prob(AnnotatedNetwork &ann_network, size_t tree_index);

pll_utree_t* displayed_tree_to_utree(Network &network, const std::vector<ReticulationState>& reticulationChoices);
pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index);
std::vector<double> collectBranchLengths(const Network &network);
void applyBranchLengths(Network &network, const std::vector<double> &branchLengths);
void setReticulationParents(Network &network, const std::vector<ReticulationState>& reticulationChoices);
void setReticulationParents(Network &network, size_t treeIdx);
/*
 * Find possible placements for the root node in a semi-rooted network.
 */
std::vector<Node*> getPossibleRootNodes(Network &network);
Node* getPossibleTreeRootNode(Network &network, const std::vector<bool> &dead_nodes);

std::vector<bool> collect_dead_nodes(Network &network, size_t megablobRootClvIndex,
        Node **displayed_tree_root = nullptr);
std::vector<Node*> grab_current_node_parents(Network &network);
std::vector<Node*> reversed_topological_sort(Network &network);

bool networkIsConnected(Network &network);

}
