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

#include <vector>

namespace netrax {

Node* getTargetNode(const Link* link);
bool isOutgoing(Node* from, Node* to);
Link* getLinkToClvIndex(Node* node, size_t target_index);
Link* getLinkToNode(Node *node, Node *target);
Node* getReticulationChild(const Node* node);
Node* getReticulationFirstParent(const Node* node);
Node* getReticulationSecondParent(const Node* node);
Node* getReticulationActiveParent(const Node *node);
double getReticulationFirstParentProb(const Node* node);
double getReticulationSecondParentProb(const Node* node);
double getReticulationActiveProb(const Node* node);
size_t getReticulationFirstParentPmatrixIndex(const Node* node);
size_t getReticulationSecondParentPmatrixIndex(const Node* node);
size_t getReticulationActiveParentPmatrixIndex(const Node* node);

std::vector<Node*> getChildren(Node* node, const Node* myParent);
std::vector<Node*> getActiveChildren(Node* node, const Node* myParent);
Node* getOtherChild(Node* parent, Node* aChild);
bool hasChild(Node* parent, Node* candidate);
std::vector<Node*> getNeighbors(const Node* node);
std::vector<Node*> getActiveNeighbors(const Node* node);
Node* getActiveParent(const Node* node);
std::vector<Node*> getAllParents(const Node* node);
Edge* getEdgeTo(const Node* node, const Node* target);
std::vector<Edge*> getAdjacentEdges(const Node* node);

Node* getSource(const Edge* edge);
Node* getTarget(const Edge* edge);

bool hasNeighbor(Node* node1, Node* node2);

Link* make_link(size_t link_id, Node *node, Edge *edge, Direction dir);

}
