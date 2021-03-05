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
#include <unordered_set>

namespace netrax {

Node* getTargetNode(Network &network, const Link *link);
bool isOutgoing(Network &network, Node *from, Node *to);
std::vector<Link*> getLinksToClvIndex(Network &network, Node *node, size_t target_index);
Link* getLinkToNode(Network &network, Node *node, Node *target);
Link* getLinkToNode(Network &network, size_t from_clv_index, size_t to_clv_index);
Node* getReticulationChild(Network &network, const Node *node);
Node* getReticulationFirstParent(Network &network, const Node *node);
Node* getReticulationSecondParent(Network &network, const Node *node);
Node* getReticulationActiveParent(Network &network, const Node *node);
Node* getReticulationOtherParent(Network &network, const Node *node, const Node *parent);
Node* getReticulationNonActiveParent(Network &network, const Node *node);
double getReticulationFirstParentProb(AnnotatedNetwork &ann_network, const Node *node);
double getReticulationSecondParentProb(AnnotatedNetwork &ann_network, const Node *node);
double getReticulationActiveProb(AnnotatedNetwork &ann_network, const Node *node);
size_t getReticulationFirstParentPmatrixIndex(const Node *node);
size_t getReticulationSecondParentPmatrixIndex(const Node *node);
size_t getReticulationActiveParentPmatrixIndex(const Node *node);

std::vector<Node*> getChildren(Network &network, Node *node);
std::vector<Node*> getActiveChildren(Network &network, Node *node);
std::vector<Node*> getActiveAliveChildren(Network &network, const std::vector<bool> &dead_nodes,
        Node *node);
std::vector<Node*> getChildrenIgnoreDirections(Network &network, Node *node, const Node *myParent);
std::vector<Node*> getActiveChildrenUndirected(Network &network, Node *node, const Node *myParent);
Node* getOtherChild(Network &network, Node *parent, Node *aChild);
bool hasChild(Network &network, Node *parent, Node *candidate);
std::vector<Node*> getNeighbors(Network &network, const Node *node);
std::vector<Node*> getActiveNeighbors(Network &network, const Node *node);
std::vector<Node*> getActiveAliveNeighbors(Network &network, const std::vector<bool> &dead_nodes,
        const Node *node);
Node* getActiveParent(Network &network, const Node *node);
std::vector<Node*> getAllParents(Network &network, const Node *node);
Edge* getEdgeTo(Network &network, const Node *node, const Node *target);
Edge* getEdgeTo(Network &network, size_t from_clv_index, size_t to_clv_index);
std::vector<Edge*> getAdjacentEdges(Network &network, const Node *node);
std::vector<Edge*> getAdjacentEdges(Network &network, const Edge *edge);

Node* getSource(Network &network, const Edge *edge);
Node* getTarget(Network &network, const Edge *edge);

bool hasNeighbor(Node *node1, Node *node2);

Link* make_link(Node *node, Edge *edge, Direction dir);

void invalidateSingleClv(pllmod_treeinfo_t *treeinfo, unsigned int clv_index);
void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, size_t partition_idx, bool invalidate_myself,
        std::vector<bool> &visited);
void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, bool invalidate_myself, std::vector<bool> &visited);
void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, size_t partition_idx, bool invalidate_myself);

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index,
        std::vector<bool> &visited);
void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index);

bool assertReticulationProbs(AnnotatedNetwork &ann_network);

std::unordered_set<size_t> getNeighborPmatrixIndices(Network &network, Edge *edge);

void setReticulationState(AnnotatedNetwork &ann_network, size_t reticulation_idx, ReticulationState state);

}
