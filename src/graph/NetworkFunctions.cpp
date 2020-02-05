/*
 * NetworkFunctions.cpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#include "NetworkFunctions.hpp"

#include <cassert>
#include <memory>
#include <iostream>
#include <stack>
#include <queue>
#include <stdexcept>

#include "Network.hpp"
#include "Edge.hpp"
#include "ReticulationData.hpp"
#include "Node.hpp"

namespace netrax {

static char* xstrdup(const char *s) {
	size_t len = strlen(s);
	char *p = (char*) malloc(len + 1);
	if (!p) {
		pll_errno = PLL_ERROR_MEM_ALLOC;
		snprintf(pll_errmsg, 200, "Memory allocation failed");
		return NULL;
	}
	return strcpy(p, s);
}

pll_unode_t* create_unode(const std::string &label) {
	pll_unode_t* new_unode = (pll_unode_t*) malloc(sizeof(pll_unode_t));
	if (!label.empty()) {
		new_unode->label = xstrdup(label.c_str());
	} else {
		new_unode->label = nullptr;
	}
	new_unode->scaler_index = -1;
	new_unode->clv_index = 0;
	new_unode->length = 0.0;
	new_unode->node_index = 0;
	new_unode->pmatrix_index = 0;
	new_unode->next = nullptr;
	return new_unode;
}

void destroy_unode(pll_unode_t* unode) {
	assert(unode);
	if (unode->label) {
		free(unode->label);
	}
	if (unode->next) {
		if (unode->next->next) {
			free(unode->next->next);
		}
		free(unode->next);
	}
	free(unode);
}

void remove_dead_children(std::vector<Node*>& children, const std::vector<bool>& dead_nodes) {
	children.erase(std::remove_if(children.begin(), children.end(), [&](Node* node) {
		assert(node);
		return (dead_nodes[node->clv_index]);
	}), children.end());
}


struct CumulatedChild {
	const Node* child = nullptr;
	const Node* direct_parent = nullptr;
	double cum_brlen = 0.0;
};

CumulatedChild getCumulatedChild(const Node* parent, const Node* child, const std::vector<bool>& dead_nodes, const std::vector<bool>& skipped_nodes) {
	CumulatedChild res{child, parent, 0.0};
	res.cum_brlen += child->getEdgeTo(parent)->length;
	const Node* act_parent = parent;
	while (skipped_nodes[res.child->clv_index]) {
		std::vector<Node*> activeChildren = res.child->getActiveChildren(act_parent);
		remove_dead_children(activeChildren, dead_nodes);
		assert(activeChildren.size() == 1);
		act_parent = res.child;
		res.child = activeChildren[0];
		res.cum_brlen += act_parent->getEdgeTo(res.child)->length;
	}
	res.direct_parent = act_parent;
	return res;
}

std::vector<CumulatedChild> getCumulatedChildren(const Node* parent, const Node* actNode, const std::vector<bool>& dead_nodes, const std::vector<bool>& skipped_nodes) {
	assert(actNode);
	std::vector<CumulatedChild> res;
	assert(!skipped_nodes[actNode->clv_index]);
	std::vector<Node*> activeChildren = actNode->getActiveChildren(parent);
	for (size_t i = 0; i < activeChildren.size(); ++i) {
		res.push_back(getCumulatedChild(actNode, activeChildren[i], dead_nodes, skipped_nodes));
	}
	return res;
}

pll_unode_t* connect_subtree_recursive(const Node *networkNode, pll_unode_t *from_parent, const Node *networkParentNode,
		const std::vector<bool>& dead_nodes, const std::vector<bool>& skipped_nodes) {
	assert(networkNode->getType() == NodeType::BASIC_NODE);

	assert(!dead_nodes[networkNode->clv_index] && !skipped_nodes[networkNode->clv_index]);

	pll_unode_t *to_parent = nullptr;
	if (networkParentNode) {
		to_parent = create_unode(networkNode->getLabel());
		to_parent->clv_index = networkNode->clv_index;
		from_parent->back = to_parent;
		to_parent->back = from_parent;
		to_parent->length = from_parent->length;
		to_parent->next = NULL;
	}

	if (networkNode->isTip()) {
		return to_parent;
	}

	std::vector<CumulatedChild> cum_children = getCumulatedChildren(networkParentNode, networkNode, dead_nodes, skipped_nodes);
	assert(cum_children.size() == 2 || (cum_children.size() == 3 && networkParentNode == nullptr)); // 2 children, or started with root node.

	std::vector<pll_unode_t*> toChildren(cum_children.size(), nullptr);
	for (size_t i = 0; i < toChildren.size(); ++i) {
		toChildren[i] = create_unode(networkNode->getLabel());
		toChildren[i]->clv_index = networkNode->clv_index;
		toChildren[i]->length = cum_children[i].cum_brlen;
		connect_subtree_recursive(cum_children[i].child, toChildren[i], cum_children[i].direct_parent, dead_nodes, skipped_nodes);
	}

	// set the next pointers
	bool isRoot = false;
	pll_unode_t *unode = to_parent;
	if (!unode) {
		unode = toChildren[0];
		isRoot = true;
	}

	for (size_t i = isRoot; i < toChildren.size(); ++i) {
		unode->next = toChildren[i];
		unode = unode->next;
	}

	unode->next = isRoot ? toChildren[0] : to_parent;

	return unode->next;
}

void fill_dead_nodes_recursive(const Node* myParent, const Node* node, std::vector<bool>& dead_nodes) {
	if (node->isTip()) {
		return;
	}
	std::vector<Node*> activeChildren = node->getActiveChildren(myParent);
	if (activeChildren.empty()) {
		dead_nodes[node->clv_index] = true;
	} else {
		for (size_t i = 0; i < activeChildren.size(); ++i) {
			fill_dead_nodes_recursive(node, activeChildren[i], dead_nodes);
		}
	}
	// count how many active children are not dead
	size_t num_undead = std::count_if(activeChildren.begin(), activeChildren.end(), [&](Node* actNode) {
		return !dead_nodes[actNode->clv_index];
	});
	if (num_undead == 0) {
		dead_nodes[node->clv_index] = true;
	}
}

void fill_skipped_nodes_recursive(const Node* myParent, const Node* node, const std::vector<bool>& dead_nodes, std::vector<bool>& skipped_nodes) {
	if (node->isTip()) {
		return; // tip nodes never need to be skipped/ contracted
	}
	std::vector<Node*> activeChildren = node->getActiveChildren(myParent);
	remove_dead_children(activeChildren, dead_nodes);

	if (activeChildren.size() < 2) {
		skipped_nodes[node->clv_index] = true;
	}

	if (activeChildren.empty()) {
		assert(dead_nodes[node->clv_index]);
	}

	for (size_t i = 0; i < activeChildren.size(); ++i) {
		fill_skipped_nodes_recursive(node, activeChildren[i], dead_nodes, skipped_nodes);
	}
}


pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index) {
	setReticulationParents(network, tree_index);
	const Node *root = nullptr;

	// find a non-reticulation node with 3 active neighbors. This will be the root of the displayed tree.
	std::vector<Node*> possibleRoots = getPossibleRootNodes(network);
	root = possibleRoots[0];
	assert(root);

	std::vector<bool> dead_nodes(network.nodes.size(), false);
	fill_dead_nodes_recursive(nullptr, root, dead_nodes);
	std::vector<bool> skipped_nodes(network.nodes.size(), false);
	fill_skipped_nodes_recursive(nullptr, root, dead_nodes, skipped_nodes);
	// now, we already know which nodes are skipped and which nodes are dead.

	pll_unode_t *uroot = connect_subtree_recursive(root, nullptr, nullptr, dead_nodes, skipped_nodes);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	pll_utree_t *utree = pll_utree_wraptree(uroot, network.num_tips());

	// ensure that the tip clv indices are the same as in the network
	for (size_t i = 0; i < utree->inner_count + utree->tip_count; ++i) {
		if (utree->nodes[i]->clv_index < utree->tip_count) {
			Node *networkNode = network.getNodeByLabel(utree->nodes[i]->label);
			utree->nodes[i]->clv_index = utree->nodes[i]->node_index = networkNode->clv_index;
		}
	}

	assert(utree->tip_count == network.num_tips());
	return utree;
}

std::vector<double> collectBranchLengths(const Network &network) {
	std::vector<double> brLengths(network.edges.size());
	for (size_t i = 0; i < network.edges.size(); ++i) {
		brLengths[i] = network.edges[i].length;
	}
	return brLengths;
}
void applyBranchLengths(Network &network, const std::vector<double> &branchLengths) {
	assert(branchLengths.size() == network.edges.size());
	for (size_t i = 0; i < network.edges.size(); ++i) {
		network.edges[i].length = branchLengths[i];
	}
}
void setReticulationParents(Network &network, size_t treeIdx) {
	for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
		// check if i-th bit is set in treeIdx
		bool activeParentIdx = treeIdx & (1 << i);
		network.reticulation_nodes[i]->getReticulationData()->setActiveParent(activeParentIdx);
	}
}

void setReticulationParents(BlobInformation& blobInfo, unsigned int megablob_idx, size_t treeIdx) {
	for (size_t i = 0; i < blobInfo.reticulation_nodes_per_megablob[megablob_idx].size(); ++i) {
		// check if i-th bit is set in treeIdx
		bool activeParentIdx = treeIdx & (1 << i);
		blobInfo.reticulation_nodes_per_megablob[megablob_idx][i]->getReticulationData()->setActiveParent(activeParentIdx);
	}
}

void forbidSubnetwork(Node *myParent, Node *node, std::vector<bool> &forbidden) {
	if (forbidden[node->getClvIndex()])
		return;
	forbidden[node->getClvIndex()] = true;
	std::vector<Node*> children = node->getChildren(myParent);
	for (size_t i = 0; i < children.size(); ++i) {
		forbidSubnetwork(node, children[i], forbidden);
	}
}

/*
 * Find possible placements for the root node in a semi-rooted network.
 */
std::vector<Node*> getPossibleRootNodes(Network &network) {
	std::vector<bool> forbidden(network.nodes.size(), false);
	for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
		forbidSubnetwork(nullptr, network.reticulation_nodes[i], forbidden);
	}
	std::vector<Node*> res;
	for (size_t i = 0; i < network.nodes.size(); ++i) {
		if (!forbidden[network.nodes[i].getClvIndex()]) {
			if (network.nodes[i].getType() == NodeType::BASIC_NODE && network.nodes[i].getActiveNeighbors().size() == 3) {
				res.push_back(&network.nodes[i]);
			}
		}
	}
	return res;
}

void getTipVectorRecursive(pll_unode_t *actParent, pll_unode_t *actNode, size_t pmatrix_idx, bool pmatrix_idx_found,
		std::vector<bool> &res) {
	if (!actNode) {
		return;
	}
	if (actNode->pmatrix_index == pmatrix_idx) {
		pmatrix_idx_found = true;
	}
	if (pllmod_utree_is_tip(actNode) && pmatrix_idx_found) {
		res[actNode->clv_index] = true;
	} else if (!pllmod_utree_is_tip(actNode)) {

		pll_unode_t *link = actNode->next;
		do {
			if (link->back != actParent) {
				getTipVectorRecursive(actNode, link->back, pmatrix_idx, pmatrix_idx_found, res);
			}
			link = link->next;
		} while (link && link != actNode);
	}
}

std::vector<bool> getTipVector(const pll_utree_t &utree, size_t pmatrix_idx) {
	std::vector<bool> res(utree.tip_count, false);
	// do a top-down preorder traversal of the tree,
	//	starting to write to the tip vector as soon as we have encountered the wanted pmatrix_idx
	getTipVectorRecursive(utree.vroot->back, utree.vroot, pmatrix_idx, false, res);

	// vroot and vroot->back have the same pmatrix index!!!
	if (utree.vroot->pmatrix_index != pmatrix_idx) {
		getTipVectorRecursive(utree.vroot, utree.vroot->back, pmatrix_idx, false, res);
	}
	return res;
}

void getTipVectorRecursive(Node *actParent, Node *actNode, size_t pmatrix_idx, bool pmatrix_idx_found, std::vector<bool> &res) {
	if ((actParent != nullptr) && (actNode->getEdgeTo(actParent)->pmatrix_index == pmatrix_idx)) {
		pmatrix_idx_found = true;
	}
	if (actNode->isTip() && pmatrix_idx_found) {
		res[actNode->clv_index] = true;
	} else if (!actNode->isTip()) {
		std::vector<Node*> activeChildren = actNode->getActiveChildren(actParent);
		for (size_t i = 0; i < activeChildren.size(); ++i) {
			getTipVectorRecursive(actNode, activeChildren[i], pmatrix_idx, pmatrix_idx_found, res);
		}
	}
}

std::vector<bool> getTipVector(const Network &network, size_t pmatrix_idx) {
	std::vector<bool> res(network.num_tips(), false);
	// do a top-down preorder traversal of the network,
	//	starting to write to the tip vector as soon as we have encountered the wanted pmatrix_idx
	getTipVectorRecursive(nullptr, network.root, pmatrix_idx, false, res);
	return res;
}

std::vector<std::vector<size_t> > getDtBranchToNetworkBranchMapping(const pll_utree_t &utree, Network &network, size_t tree_idx) {
	std::vector<std::vector<size_t> > res(utree.edge_count);
	setReticulationParents(network, tree_idx);

	// for each branch, we need to figure out which tips are on one side of the branch, and which tips are on the other side
	// so essentially, we need to compare bipartitions. That's all!

	// ... we can easily get the set of tips which are in a subtree!
	//  (of either of the endpoints of the current branch, we don't really care)!!!

	// and we can use a bool vector for all tips...

	std::vector<std::vector<bool> > networkTipVectors(network.num_branches());
	for (size_t i = 0; i < network.num_branches(); ++i) {
		networkTipVectors[network.edges[i].pmatrix_index] = getTipVector(network, network.edges[i].pmatrix_index);
	}

	for (size_t i = 0; i < utree.edge_count; ++i) {
		std::vector<bool> tipVecTree = getTipVector(utree, i);
		for (size_t j = 0; j < network.num_branches(); ++j) {
			std::vector<bool> tipVecNetwork = networkTipVectors[network.edges[j].pmatrix_index];
			if (tipVecTree == tipVecNetwork) {
				res[i].push_back(network.edges[j].pmatrix_index);
			} else {
				// check if they are all different
				bool allDifferent = true;
				for (size_t k = 0; k < tipVecTree.size(); ++k) {
					if (tipVecTree[k] == tipVecNetwork[k]) {
						allDifferent = false;
					}
				}
				if (allDifferent) {
					res[i].push_back(network.edges[j].pmatrix_index);
				}
			}
		}
	}
	return res;
}

void grab_current_node_parents_recursive(std::vector<Node*>& parent, Node* actNode) {
	assert(actNode != nullptr);
	std::vector<Node*> children = actNode->getChildren(parent[actNode->clv_index]);
	for (Node* child : children){
		assert(child != nullptr);
		parent[child->clv_index] = actNode;
		grab_current_node_parents_recursive(parent, child);
	}
}

std::vector<Node*> grab_current_node_parents(const Network& network) {
	std::vector<Node*> parent(network.num_nodes(), nullptr);
	grab_current_node_parents_recursive(parent, network.root);
	return parent;
}

std::vector<Node*> reversed_topological_sort(const Network& network) {
	std::vector<Node*> res;
	res.reserve(network.num_nodes());
	std::vector<Node*> parent = grab_current_node_parents(network);
	std::vector<unsigned int> indeg(network.num_nodes(), 0);

	std::queue<Node*> q;

	// Kahn's algorithm for topological sorting

	// compute indegree of all nodes
	for (size_t i = 0; i < network.num_nodes(); ++i) {
		Node* actNode = (Node*) &(network.nodes[i]); // TODO: dirty hack, trying to make pointer non-const
		size_t act_clv_idx = actNode->clv_index;
		indeg[act_clv_idx] = actNode->getChildren(parent[act_clv_idx]).size();
		if (indeg[act_clv_idx] == 0) {
			q.emplace(actNode);
		}
	}

	size_t num_visited_vertices = 0;
	while (!q.empty()) {
		Node* actNode = q.front();
		q.pop();
		res.emplace_back(actNode);

		for (Node* neigh : actNode->getChildren(parent[actNode->clv_index])) {
			indeg[neigh->clv_index]--;
			if (indeg[neigh->clv_index] == 0) {
				q.emplace(neigh);
			}
		}
		num_visited_vertices++;
	}

	if (num_visited_vertices != network.num_nodes()) {
		throw std::runtime_error("Cycle in network detected");
	}

	return res;
}


}
