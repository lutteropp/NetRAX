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

pll_unode_t* create_unode(const std::string& label) {
	if (label != "") {
		return pllmod_utree_create_node(0, 0, xstrdup(label.c_str()), NULL);
	} else {
		return pllmod_utree_create_node(0, 0, NULL, NULL);
	}
}

void make_connections(const Node *networkNode, pll_unode_t *unode) {
	assert(networkNode->getType() == NodeType::BASIC_NODE);
	unode->next = NULL;

	Node *networkParentNode = networkNode->getLink()->getTargetNode();

	std::vector<Node*> children = networkNode->getActiveChildren(networkParentNode);
	double length_to_add = 0;

	const Node *childParentNode = networkNode;
	while (children.size() == 1) { // this is the case if one of the children is a reticulation node but it's not active
		// in this case, we need to skip the other child node and directly connect to the next
		length_to_add += children[0]->getEdgeTo(childParentNode)->length;
		const Node *newChildParentNode = children[0];
		children = children[0]->getActiveChildren(childParentNode);
		childParentNode = newChildParentNode;
	}

	// now we should have either zero children (leaf node), or 2 children (inner tree node)
	assert(children.empty() || children.size() == 2);
	if (!children.empty()) {
		// TODO: if a passive child is a reticulation node, we need to do some special case stuff
		double child1LenToAdd = 0.0;
		const Node* child1ParentNode = childParentNode;
		while (children[0]->getType() == NodeType::RETICULATION_NODE) {
			child1LenToAdd += children[0]->getEdgeTo(child1ParentNode)->length;
			Node *reticulationChild = children[0]->getReticulationData()->getLinkToChild()->getTargetNode();
			if (reticulationChild->getType() == NodeType::RETICULATION_NODE
					&& reticulationChild->getReticulationData()->getLinkToActiveParent()->getTargetNode() != children[0]) {
				children[0] = nullptr;
				break;
			} else {
				child1ParentNode = children[0];
				children[0] = reticulationChild;
			}
		}

		while (children[0]->getActiveChildren(child1ParentNode).size() == 1) {
			Node* temp = children[0];
			child1LenToAdd += children[0]->getEdgeTo(child1ParentNode)->length;
			children[0] = children[0]->getActiveChildren(child1ParentNode)[0];
			child1ParentNode = temp;
		}

		double child2LenToAdd = 0.0;
		const Node* child2ParentNode = childParentNode;
		while (children[1]->getType() == NodeType::RETICULATION_NODE) {
			child2LenToAdd += children[1]->getEdgeTo(child2ParentNode)->length;
			Node *reticulationChild = children[1]->getReticulationData()->getLinkToChild()->getTargetNode();
			if (reticulationChild->getType() == NodeType::RETICULATION_NODE
					&& reticulationChild->getReticulationData()->getLinkToActiveParent()->getTargetNode() != children[1]) {
				children[1] = nullptr;
				break;
			} else {
				child2ParentNode = children[1];
				children[1] = reticulationChild;
			}
		}

		while (children[1]->getActiveChildren(child2ParentNode).size() == 1) {
			Node* temp = children[1];
			child2LenToAdd += children[1]->getEdgeTo(child2ParentNode)->length;
			children[1] = children[1]->getActiveChildren(child2ParentNode)[0];
			child2ParentNode = temp;
		}

		assert(children[0]);
		assert(children[1]);

		assert(children[0]->getType() == NodeType::BASIC_NODE);
		assert(children[1]->getType() == NodeType::BASIC_NODE);

		pll_unode_t *fromChild1 = create_unode(children[0]->getLabel());
		fromChild1->length = children[0]->getEdgeTo(child1ParentNode)->length + length_to_add + child1LenToAdd;
		pll_unode_t *toChild1 = create_unode(networkNode->getLabel());
		toChild1->length = children[0]->getEdgeTo(child1ParentNode)->length + length_to_add + child1LenToAdd;
		toChild1->back = fromChild1;
		fromChild1->back = toChild1;

		pll_unode_t *fromChild2 = create_unode(children[1]->getLabel());
		fromChild2->length = children[1]->getEdgeTo(child2ParentNode)->length + length_to_add + child2LenToAdd;
		pll_unode_t *toChild2 = create_unode(networkNode->getLabel());
		toChild2->length = children[1]->getEdgeTo(child2ParentNode)->length + length_to_add + child2LenToAdd;
		toChild2->back = fromChild2;
		fromChild2->back = toChild2;

		unode->next = toChild1;
		unode->next->next = toChild2;
		unode->next->next->next = unode;
		make_connections(children[0], fromChild1);
		make_connections(children[1], fromChild2);
	}
}

pll_utree_t* handleRootPassiveReticulation(Network &network) {
	Node *root = network.root;
	Node *root_back = network.root->getLink()->getTargetNode();
	assert(root->getType() == NodeType::BASIC_NODE);
	assert(root_back->getType() == NodeType::RETICULATION_NODE);
	assert(root_back->getReticulationData()->getLinkToActiveParent()->getTargetNode() != root);

	// root is not the active parent of root_back
	// this is a more difficult case, as also the root changes

	Node *new_root;
	Node *other_child;
	// we need to get a non-leaf active child node as the new root.
	std::vector<Node*> activeChildren = root->getActiveChildren(root_back);
	assert(activeChildren.size() == 2);
	if (activeChildren[0]->isTip()) {
		assert(!activeChildren[1]->isTip());
		new_root = activeChildren[1];
		other_child = activeChildren[0];
	} else {
		new_root = activeChildren[0];
		other_child = activeChildren[1];
	}

	pll_unode_t *uroot = create_unode(new_root->getLabel());
	pll_unode_t *uroot_back = create_unode(other_child->getLabel());

	double edgeLen = new_root->getLink()->edge->length + other_child->getLink()->edge->length;
	uroot->length = edgeLen;
	uroot_back->length = edgeLen;
	uroot->back = uroot_back;
	uroot_back->back = uroot;

	make_connections(new_root, uroot);
	make_connections(other_child, uroot_back);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

pll_utree_t* handleRootActiveReticulation(Network &network) {
	Node *root = network.root;
	Node *root_back = network.root->getLink()->getTargetNode();
	assert(root->getType() == NodeType::BASIC_NODE);
	assert(root_back->getType() == NodeType::RETICULATION_NODE);
	pll_unode_t *uroot = create_unode(root->getLabel());

	// skip the reticulation node on its way...
	double skippedLen = 0.0;
	while (root_back->getType() == NodeType::RETICULATION_NODE) {
		skippedLen += root_back->getLink()->edge->length;
		root_back = root_back->getReticulationData()->getLinkToChild()->getTargetNode();
	}
	double totalLen = skippedLen + root_back->getLink()->edge->length;
	pll_unode_t *uroot_back = create_unode(root_back->getLabel());
	uroot->back = uroot_back;
	uroot_back->back = uroot;
	uroot->length = totalLen;
	uroot_back->length = totalLen;
	make_connections(root, uroot);
	make_connections(root_back, uroot_back);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

pll_utree_t* handleRootNormal(Network& network, Node* root) {
	Node* root_back = root->getLink()->getTargetNode();

	if (root->getActiveChildren(root_back).size() == 1) { // Problem: The root is an inactive parent of a reticulation node
		if (!root_back->isTip()) {
			return handleRootNormal(network, root_back);
		} else {
			return handleRootNormal(network, root->getActiveChildren(root_back)[0]);
		}
	}

	pll_unode_t *uroot = create_unode(root->getLabel());
	pll_unode_t *uroot_back = create_unode(root_back->getLabel());
	uroot->length = root->getLink()->edge->length;
	uroot_back->length = root_back->getLink()->edge->length;
	uroot->back = uroot_back;
	uroot_back->back = uroot;
	make_connections(root, uroot);
	make_connections(root_back, uroot_back);
	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

Node* getCollapsedChild(Node* child, const Node** parent, double* cumulated_length) {
	while (child->getActiveChildren(*parent).size() == 1) {
		const Node* temp = child;
		*cumulated_length += child->getEdgeTo(*parent)->length;
		child = child->getActiveChildren(*parent)[0];
		*parent = temp;
	}
	return child;
}

pll_unode_t* connect_subtree_recursive(const Node* networkNode, pll_unode_t* from_parent, const Node *networkParentNode) {
	assert(networkNode->getType() == NodeType::BASIC_NODE);

	pll_unode_t* to_parent = nullptr;
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
		const Node* myParent = childParentNode;
		children[i] = getCollapsedChild(children[i], &myParent, &childLengths[i]);
		toChildren[i]->length = childLengths[i] + children[i]->getEdgeTo(myParent)->length;
		connect_subtree_recursive(children[i], toChildren[i], myParent);
	}

	// set the next pointers
	bool isRoot = false;
	pll_unode_t* unode = to_parent;
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
	std::vector<const Node*> possibleRoots = getPossibleRootNodes(network);
	for (size_t i = 0; i < possibleRoots.size(); ++i) {
		if (!possibleRoots[i]->isTip() && possibleRoots[i]->getType() == NodeType::BASIC_NODE) {
			if (possibleRoots[i]->getActiveNeighbors().size() == 3) {
				root = possibleRoots[i];
				break;
			}
		}
	}
	assert(root);

	pll_unode_t* uroot = connect_subtree_recursive(root, nullptr, nullptr);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

std::vector<double> collectBranchLengths(const Network& network) {
	std::vector<double> brLengths(network.edges.size());
	for (size_t i = 0; i < network.edges.size(); ++i) {
		brLengths[i] = network.edges[i].length;
	}
	return brLengths;
}
void applyBranchLengths(Network& network, const std::vector<double>& branchLengths) {
	assert(branchLengths.size() == network.edges.size());
	for (size_t i = 0; i < network.edges.size(); ++i) {
		network.edges[i].length = branchLengths[i];
	}
}
void setReticulationParents(Network& network, size_t treeIdx) {
	for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
		// check if i-th bit is set in treeIdx
		bool activeParentIdx = treeIdx & (1 << i);
		network.reticulation_nodes[i]->getReticulationData()->setActiveParent(activeParentIdx);
	}
}

void forbidSubnetwork(Node* myParent, Node* node, std::vector<bool>& forbidden) {
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
std::vector<const Node*> getPossibleRootNodes(const Network& network) {
	std::vector<bool> forbidden(network.nodes.size(), false);
	for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
		forbidSubnetwork(nullptr, network.reticulation_nodes[i], forbidden);
	}
	std::vector<const Node*> res;
	for (size_t i = 0; i < network.nodes.size(); ++i) {
		if (!forbidden[network.nodes[i].getClvIndex()]) {
			res.push_back(&network.nodes[i]);
		}
	}
	return res;
}

}
