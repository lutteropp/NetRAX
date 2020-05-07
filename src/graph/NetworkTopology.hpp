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
#include "AnnotatedNetwork.hpp"

#include <vector>

namespace netrax {

Node* getTargetNode(Network& network, const Link* link);
bool isOutgoing(Network &network, Node* from, Node* to);
Link* getLinkToClvIndex(Network &network, Node* node, size_t target_index);
Link* getLinkToNode(Network &network, Node *node, Node *target);
Link* getLinkToNode(Network &network, size_t from_clv_index, size_t to_clv_index);
Node* getReticulationChild(Network &network, const Node* node);
Node* getReticulationFirstParent(Network &network, const Node* node);
Node* getReticulationSecondParent(Network &network, const Node* node);
Node* getReticulationActiveParent(Network &network, const Node *node);
double getReticulationFirstParentProb(Network &network, const Node* node);
double getReticulationSecondParentProb(Network &network, const Node* node);
double getReticulationActiveProb(Network &network, const Node* node);
size_t getReticulationFirstParentPmatrixIndex(Network &network, const Node* node);
size_t getReticulationSecondParentPmatrixIndex(Network &network, const Node* node);
size_t getReticulationActiveParentPmatrixIndex(Network &network, const Node* node);

std::vector<Node*> getChildren(Network &network, Node* node, const Node* myParent);
std::vector<Node*> getActiveChildren(Network &network, Node* node, const Node* myParent);
Node* getOtherChild(Network &network, Node* parent, Node* aChild);
bool hasChild(Network &network, Node* parent, Node* candidate);
std::vector<Node*> getNeighbors(Network &network, const Node* node);
std::vector<Node*> getActiveNeighbors(Network &network, const Node* node);
Node* getActiveParent(Network &network, const Node* node);
std::vector<Node*> getAllParents(Network &network, const Node* node);
Edge* getEdgeTo(Network &network, const Node* node, const Node* target);
Edge* getEdgeTo(Network &network, size_t from_clv_index, size_t to_clv_index);
std::vector<Edge*> getAdjacentEdges(Network &network, const Node* node);

Node* getSource(Network &network, const Edge* edge);
Node* getTarget(Network &network, const Edge* edge);

bool hasNeighbor(Node* node1, Node* node2);

Link* make_link(Node *node, Edge *edge, Direction dir);

void invalidateHigherClvs(Network &network, pllmod_treeinfo_t *treeinfo, std::vector<bool> &visited, Node *node, bool invalidate_myself = false);
void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, bool invalidate_myself = false);

}
