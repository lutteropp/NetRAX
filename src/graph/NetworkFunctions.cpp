/*
 * NetworkFunctions.cpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#include "NetworkFunctions.hpp"

#include <cassert>
#include <memory>

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
	if (label != "") {
		return pllmod_utree_create_node(0, 0, xstrdup(label.c_str()), NULL);
	} else {
		return pllmod_utree_create_node(0, 0, NULL, NULL);
	}
}

Node* getCollapsedChild(Node *child, const Node **parent, double *cumulated_length) {
	while (child->getActiveChildren(*parent).size() == 1) {
		const Node *temp = child;
		*cumulated_length += child->getEdgeTo(*parent)->length;
		child = child->getActiveChildren(*parent)[0];
		*parent = temp;
	}
	return child;
}

pll_unode_t* connect_subtree_recursive(const Node *networkNode, pll_unode_t *from_parent,
		const Node *networkParentNode) {
	assert(networkNode->getType() == NodeType::BASIC_NODE);

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

	std::vector<Node*> children = networkNode->getActiveChildren(networkParentNode);

	// skip children that themselves have only one active child
	double length_to_add = 0;
	const Node *childParentNode = networkNode;
	while (children.size() == 1) { // this is the case if one of the children is a reticulation node but it's not active
		// in this case, we need to skip the other child node and directly connect to the next
		length_to_add += children[0]->getEdgeTo(childParentNode)->length;
		const Node *newChildParentNode = children[0];
		children = children[0]->getActiveChildren(childParentNode);
		childParentNode = newChildParentNode;
	}

	assert(children.size() == 2 || (children.size() == 3 && networkParentNode == nullptr)); // 2 children, or started with root node.

	std::vector<pll_unode_t*> toChildren(children.size(), nullptr);
	for (size_t i = 0; i < toChildren.size(); ++i) {
		toChildren[i] = create_unode(networkNode->getLabel());
		toChildren[i]->clv_index = networkNode->clv_index;
	}

	std::vector<double> childLengths(children.size(), length_to_add);
	for (size_t i = 0; i < children.size(); ++i) {
		const Node *myParent = childParentNode;
		children[i] = getCollapsedChild(children[i], &myParent, &childLengths[i]);
		toChildren[i]->length = childLengths[i] + children[i]->getEdgeTo(myParent)->length;
		connect_subtree_recursive(children[i], toChildren[i], myParent);
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

pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index) {
	setReticulationParents(network, tree_index);
	const Node *root = nullptr;

	// find a non-reticulation node with 3 active neighbors. This will be the root of the displayed tree.
	std::vector<Node*> possibleRoots = getPossibleRootNodes(network);
	root = possibleRoots[0];
	assert(root);

	pll_unode_t *uroot = connect_subtree_recursive(root, nullptr, nullptr);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	pll_utree_t *utree = pll_utree_wraptree(uroot, network.num_tips());

	// ensure that the tip clv indices are the same as in the network
	for (size_t i = 0; i < utree->inner_count + utree->tip_count; ++i) {
		if (utree->nodes[i]->clv_index < utree->tip_count) {
			Node *networkNode = network.getNodeByLabel(utree->nodes[i]->label);
			utree->nodes[i]->clv_index = utree->nodes[i]->node_index = networkNode->clv_index;
		}
	}

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
			if (network.nodes[i].getType() == NodeType::BASIC_NODE
					&& network.nodes[i].getActiveNeighbors().size() == 3) {
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
		pll_unode_t *link = actNode;
		do {
			if (link->back != actParent) {
				getTipVectorRecursive(actNode, link->back, pmatrix_idx, pmatrix_idx_found, res);
			}
			link = link->next;
		} while (link != actNode);
	}
}

std::vector<bool> getTipVector(const pll_utree_t &utree, size_t pmatrix_idx) {
	std::vector<bool> res(utree.tip_count, false);
	// do a top-down preorder traversal of the tree,
	//	starting to write to the tip vector as soon as we have encountered the wanted pmatrix_idx
	getTipVectorRecursive(nullptr, utree.vroot, pmatrix_idx, false, res);

	return res;
}

void getTipVectorRecursive(Node *actParent, Node *actNode, size_t pmatrix_idx, bool pmatrix_idx_found,
		std::vector<bool> &res) {
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

std::vector<size_t> getDtBranchToNetworkBranchMapping(const pll_utree_t &utree, Network &network, size_t tree_idx) {
	std::vector<size_t> res(utree.edge_count);
	setReticulationParents(network, tree_idx);

	// for each branch, we need to figure out which tips are on one side of the branch, and which tips are on the other side
	// so essentially, we need to compare bipartitions. That's all!

	// ... we can easily get the set of tips which are in a subtree!
	//  (of either of the endpoints of the current branch, we don't really care)!!!

	// and we can use a bool vector for all tips...

	std::vector<std::vector<bool> > networkTipVectors(network.num_branches());
	for (size_t i = 0; i < network.num_branches(); ++i) {
		networkTipVectors[i] = getTipVector(network, i);
	}

	for (size_t i = 0; i < utree.edge_count; ++i) {
		std::vector<bool> tipVecTree = getTipVector(utree, i);
		bool found = false;
		for (size_t j = 0; j < network.num_branches(); ++j) {
			std::vector<bool> tipVecNetwork = getTipVector(network, j);
			if (tipVecTree == tipVecNetwork) {
				res[i] = j;
				found = true;
				break;
			} else {
				// check if they are all different
				bool allDifferent = true;
				for (size_t k = 0; k < tipVecTree.size(); ++k) {
					if (tipVecTree[k] == tipVecNetwork[k]) {
						allDifferent = false;
						break;
					}
				}
				if (allDifferent) {
					res[i] = j;
					found = true;
					break;
				}
			}
		}
		assert(found);
	}
	return res;
}

}
