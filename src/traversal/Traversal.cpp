/*
 * Traversal.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "Traversal.hpp"

#include "../Network.hpp"

namespace netrax {

void postorder(Node* parent, Node* actNode, std::vector<Node*>& buffer) {
	if (actNode->type == NodeType::BASIC_NODE) { // visit the two children
		std::vector<Node*> neighbors = actNode->getNeighbors();
		for (size_t i = 0; i < neighbors.size(); ++i) {
			if (neighbors[i] != parent) {
				// check if the current neighbor is a reticulation node, if so, only go this path if we are the active parent
				if (neighbors[i]->type != NodeType::RETICULATION_NODE || neighbors[i]->getReticulationData()->getLinkToActiveParent()->getTargetNode() == actNode) {
					postorder(actNode, neighbors[i], buffer);
				}
			}
		}
	} else { // only visit the child node
		postorder(actNode, actNode->getReticulationData()->getLinkToChild()->getTargetNode(), buffer);
	}

	// 2) visit the node itself
	buffer.push_back(actNode);
}

std::vector<Node*> postorderTraversal(Network& network, size_t tree_index) {
	std::vector<Node*> buffer;
	network.setReticulationParents(tree_index);
	postorder(nullptr, network.root, buffer);
	return buffer;
}

}
