/*
 * Helper.hpp
 *
 *  Created on: Apr 16, 2020
 *      Author: sarah
 */

#pragma once

#include "../graph/Node.hpp"
#include "../graph/Link.hpp"
#include "../graph/Edge.hpp"
#include "../graph/Network.hpp"
#include "../graph/ReticulationData.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/DisplayedTreeData.hpp"

#include <vector>
#include <unordered_set>
#include <utility>

namespace netrax {

/* helper functions related to reticulations (ReticulationHelper.cpp) */
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
size_t getReticulationChildPmatrixIndex(const Node* node);
bool assertReticulationProbs(AnnotatedNetwork &ann_network);
void setReticulationState(Network &network, size_t reticulation_idx, ReticulationState state);
void setReticulationParents(Network &network, const std::vector<ReticulationState>& reticulationChoices);
void setReticulationParents(Network &network, size_t treeIdx);

/* helper functions related to links (LinkHelper.cpp) */
Node* getTargetNode(const Network &network, const Link *link);
Link* make_link(Node *node, Edge *edge, Direction dir);
std::vector<const Link*> getLinksToClvIndex(Network &network, const Node *node, size_t target_index);
std::vector<Link*> getLinksToClvIndexMutable(Network &network, Node *node, size_t target_index);
Link* getLinkToNode(Network &network, Node *node, Node *target);
Link* getLinkToNode(Network &network, size_t from_clv_index, size_t to_clv_index);

/* helper functions related to edges (EdgeHelper.cpp) */
Edge* getEdgeTo(Network &network, const Node *node, const Node *target);
Edge* getEdgeTo(Network &network, size_t from_clv_index, size_t to_clv_index);
std::vector<Edge*> getAdjacentEdges(Network &network, const Node *node);
std::vector<Edge*> getAdjacentEdges(Network &network, const Edge *edge);
Node* getSource(Network &network, const Edge *edge);
Node* getTarget(Network &network, const Edge *edge);
bool isOutgoing(Network &network, const Node *from, const Node *to);
bool isActiveBranch(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, unsigned int pmatrix_index);
bool isActiveAliveBranch(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, const std::vector<bool>& dead_nodes, unsigned int pmatrix_index);
bool isActiveAliveBranch(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, unsigned int pmatrix_index);

/* helper functions related to children (ChildrenHelper.cpp) */
std::vector<Node*> getChildren(Network &network, const Node *node);
std::vector<Node*> getActiveChildren(Network &network, const Node *node);
std::vector<Node*> getActiveAliveChildren(Network &network, const std::vector<bool> &dead_nodes,
        const Node *node);
std::vector<Node*> getChildrenIgnoreDirections(Network &network, const Node *node, const Node *myParent);
std::vector<Node*> getActiveChildrenUndirected(Network &network, const Node *node, const Node *myParent);
Node* getOtherChild(Network &network, const Node *parent, const Node *aChild);
bool hasChild(Network &network, const Node *parent, const Node *candidate);
std::vector<Node*> getCurrentChildren(AnnotatedNetwork& ann_network, const Node* node, const Node* parent, const ReticulationConfigSet& restrictions);

/* helper functions related to parents (ParentHelper.cpp) */
Node* getActiveParent(Network &network, const Node *node);
std::vector<Node*> getAllParents(Network &network, const Node *node);
std::vector<Node*> getParentPointers(AnnotatedNetwork& ann_network, const Node* virtual_root);
std::vector<Node*> getParentPointers(AnnotatedNetwork& ann_network, const std::vector<ReticulationState>& reticulationChoices, const Node* virtual_root);
std::vector<Node*> grab_current_node_parents(Network &network);

/* helper functions related to neighbors (NeighborHelper.cpp) */
std::vector<Node*> getNeighbors(const Network &network, const Node *node);
std::vector<Node*> getActiveNeighbors(Network &network, const Node *node);
std::vector<Node*> getActiveAliveNeighbors(Network &network, const std::vector<bool> &dead_nodes,
        const Node *node);
bool hasNeighbor(const Node *node1, const Node *node2);
std::unordered_set<size_t> getNeighborPmatrixIndices(Network &network, const Edge *edge);
std::vector<Node*> getNeighborsWithinRadius(const Network& network, const Node* node, int min_radius, int max_radius);
bool topology_equal(Network& n1, Network& n2);

/* helper functions related to clv/pmatrix invalidation (InvalidationHelper.cpp) */
void validateSingleClv(AnnotatedNetwork& ann_network, unsigned int clv_index);
void invalidateHigherCLVs(AnnotatedNetwork &ann_network, const Node *node, bool invalidate_myself,
        std::vector<bool> &visited);
void invalidateHigherCLVs(AnnotatedNetwork &ann_network, const Node *node, bool invalidate_myself, std::vector<bool> &visited);
void invalidateHigherCLVs(AnnotatedNetwork &ann_network, const Node *node, bool invalidate_myself);
void invalidateHigherPseudoCLVs(AnnotatedNetwork &ann_network, const Node *node, bool invalidate_myself);
void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index,
        std::vector<bool> &visited);
void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index);
void invalidPmatrixIndexOnly(AnnotatedNetwork& ann_network, size_t pmatrix_index);
bool allClvsValid(pllmod_treeinfo_t* treeinfo, size_t clv_index);
void invalidate_pmatrices(AnnotatedNetwork &ann_network,
        std::vector<size_t> &affectedPmatrixIndices);
void invalidateAllCLVs(AnnotatedNetwork &ann_network);


/* helper functions related to reticulation config set (ReticulationConfigHelper.cpp) */
ReticulationConfigSet getRestrictionsToDismissNeighbor(AnnotatedNetwork& ann_network, const  Node* node, const  Node* neighbor);
ReticulationConfigSet getRestrictionsToTakeNeighbor(AnnotatedNetwork& ann_network, const  Node* node, const  Node* neighbor);
ReticulationConfigSet getReticulationChoicesThisOnly(AnnotatedNetwork& ann_network, const ReticulationConfigSet& this_tree_config, const ReticulationConfigSet& other_child_dead_settings, const  Node* parent, const  Node* this_child, const  Node* other_child);
ReticulationConfigSet deadNodeSettings(AnnotatedNetwork& ann_network, const NodeDisplayedTreeData& displayed_trees, const Node* parent, const Node* child);
ReticulationConfigSet getTreeConfig(AnnotatedNetwork& ann_network, size_t tree_idx);
DisplayedTreeData& findMatchingDisplayedTree(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, NodeDisplayedTreeData& data);
Node* findFirstNodeWithTwoActiveChildren(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, const Node* oldRoot);
const TreeLoglData& getMatchingTreeData(const std::vector<DisplayedTreeData>& trees, const ReticulationConfigSet& queryChoices);
ReticulationConfigSet getRestrictionsActiveBranch(AnnotatedNetwork& ann_network, size_t pmatrix_index);
ReticulationConfigSet getRestrictionsActiveAliveBranch(AnnotatedNetwork& ann_network, size_t pmatrix_index);

}