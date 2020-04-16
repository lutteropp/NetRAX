/*
 * NetworkTopology.hpp
 *
 *  Created on: Apr 16, 2020
 *      Author: sarah
 */

#pragma once

#include "Node.hpp"
#include "Link.hpp"
#include "Edge.hpp"
#include "Network.hpp"
#include "ReticulationData.hpp"

namespace netrax {

Node* getTargetNode(const Link* link);
Link* getLinkToClvIndex(Node* node, size_t target_index);
Node* getReticulationChild(const Node* node);
Node* getReticulationFirstParent(const Node* node);
Node* getReticulationSecondParent(const Node* node);
Node* getReticulationActiveParent(const Node *node);

std::vector<Node*> getChildren(const Node* node, const Node* myParent);
std::vector<Node*> getActiveChildren(const Node* node, const Node* myParent);
std::vector<Node*> getNeighbors(const Node* node);
std::vector<Node*> getActiveNeighbors(const Node* node);
Node* getActiveParent(const Node* node);
std::vector<Node*> getAllParents(const Node* node);
Edge* getEdgeTo(const Node* node, const Node* target);

}
