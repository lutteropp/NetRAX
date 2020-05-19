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
namespace netrax {

struct Node;
class Network;
struct BlobInformation;

std::vector<std::vector<size_t> > getDtBranchToNetworkBranchMapping(const pll_utree_t &utree, Network &network,
        size_t tree_idx);

pll_utree_t* displayed_tree_to_utree(Network &network, const std::vector<Node*> &travbuffer, size_t tree_index);
std::vector<double> collectBranchLengths(const Network &network);
void applyBranchLengths(Network &network, const std::vector<double> &branchLengths);
void setReticulationParents(Network &network, size_t treeIdx);
void setReticulationParents(BlobInformation &blobInfo, unsigned int megablob_idx, size_t treeIdx);
/*
 * Find possible placements for the root node in a semi-rooted network.
 */
std::vector<Node*> getPossibleRootNodes(Network &network);
Node* getPossibleTreeRootNode(Network &network, const std::vector<bool> &dead_nodes);

std::vector<bool> collect_dead_nodes(Network &network, Node** displayed_tree_root = nullptr);
std::vector<Node*> grab_current_node_parents(Network &network);
std::vector<Node*> reversed_topological_sort(Network &network);

std::string exportDebugInfoBlobs(Network &network, const BlobInformation &blobInfo);
std::string exportDebugInfoExtraNodeNumber(Network &network, const std::vector<unsigned int> &extra_node_number);
std::string exportDebugInfo(Network &network);

bool networkIsConnected(Network &network);

}
