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
#include "../graph/DisplayedTreeData.hpp"

namespace netrax {

struct Node;
class Network;
class AnnotatedNetwork;

double displayed_tree_prob(AnnotatedNetwork &ann_network, size_t tree_index);
pll_utree_t* displayed_tree_to_utree(Network &network, const std::vector<ReticulationState>& reticulationChoices);
pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index);

std::vector<bool> collect_dead_nodes(Network &network, size_t rootClvIndex,
        Node **displayed_tree_root = nullptr);
std::vector<Node*> reversed_topological_sort(Network &network);
bool networkIsConnected(Network &network);

}
