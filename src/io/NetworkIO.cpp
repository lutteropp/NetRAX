/*
 * NetworkConverter.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "NetworkIO.hpp"

#include <stddef.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../graph/Direction.hpp"
#include "../graph/Edge.hpp"
#include "../graph/Link.hpp"
#include "../graph/Network.hpp"
#include "../graph/Node.hpp"
#include "../graph/NodeType.hpp"
#include "../graph/ReticulationData.hpp"

namespace netrax {

Link* buildBackLink(Link *myLink, const RootedNetworkNode *targetNode, const RootedNetworkNode *parentNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index, size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network, std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations);

void setPMatrixIndexConditional(Edge *edge, size_t *inner_pmatrix_index) {
	if (edge->link1->node->isTip()) {
		edge->pmatrix_index = edge->link1->node->clv_index;
	} else if (edge->link2->node->isTip()) {
		edge->pmatrix_index = edge->link2->node->clv_index;
	} else {
		edge->pmatrix_index = (*inner_pmatrix_index)++;
	}
}

Link* buildBackLinkReticulationFirstVisit(Link *myLink, const RootedNetworkNode *targetNode, const RootedNetworkNode *parentNode,
		size_t *inner_clv_index, size_t *inner_scaler_index, size_t *inner_pmatrix_index, size_t *inner_node_index, size_t *actNodeCount,
		size_t *actEdgeCount, size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	assert(targetNode->isReticulation);

	double firstEncounteredParentProb;
	if (targetNode->firstParent == parentNode) {
		firstEncounteredParentProb = targetNode->firstParentProb;
	} else {
		assert(targetNode->secondParent == parentNode);
		firstEncounteredParentProb = targetNode->secondParentProb;
	}
	// firstLink goes to first encountered parent, secondLink goes to second encountered parent, third link goes to child.

	Node *unode = &network.nodes[(*actNodeCount)++];
	Link *firstLink = &network.links[(*actLinkCount)++];
	Link *secondLink = &network.links[(*actLinkCount)++];
	Link *thirdLink = &network.links[(*actLinkCount)++];
	firstLink->next = secondLink;
	secondLink->next = thirdLink;
	thirdLink->next = firstLink;
	firstLink->node = unode;
	secondLink->node = unode;
	thirdLink->node = unode;

	Edge *firstEdge = myLink->edge;
	// delay creation of second edge, as we haven't yet encountered the second parent
	Edge *thirdEdge = &network.edges[(*actEdgeCount)++];
	//firstEdge->link1 is already set
	assert(firstEdge->link1 == myLink);
	thirdEdge->link1 = thirdLink;

	// firstEdge->length is already set
	assert(firstEdge->length == targetNode->firstParentLength);
	thirdEdge->length = targetNode->children[0]->length;
	thirdEdge->support = targetNode->children[0]->support;

	firstLink->edge = firstEdge;
	secondLink->edge = nullptr; // will be set later
	thirdLink->edge = thirdEdge;

	Link *link1Back = myLink;
	Link *link2Back = nullptr; // will be set later
	Link *link3Back = buildBackLink(thirdLink, targetNode->children[0], targetNode, inner_clv_index, inner_scaler_index,
			inner_pmatrix_index, inner_node_index, actNodeCount, actEdgeCount, actLinkCount, network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	// firstEdge->link2 will be set by the caller
	thirdEdge->link2 = link3Back;

	ReticulationData retData;
	retData.init(targetNode->reticulationId, targetNode->reticulationName, 0, firstLink, secondLink, thirdLink, firstEncounteredParentProb);
	unode->initReticulation((*inner_clv_index)++, (*inner_scaler_index)++, firstLink, targetNode->label, retData);

	// set the indices now
	firstLink->node_index = (*inner_node_index)++;
	secondLink->node_index = (*inner_node_index)++;
	thirdLink->node_index = (*inner_node_index)++;
	// firstEdge->pmatrix_index is already set
	setPMatrixIndexConditional(thirdEdge, inner_pmatrix_index);

	network.inner_nodes.push_back(unode);
	network.reticulation_nodes.push_back(unode);
	visitedReticulations[targetNode] = unode;

	assert(firstEdge->length != 0);
	assert(thirdEdge->length != 0);

	return firstLink;
}

Link* buildBackLinkReticulationSecondVisit(Link *myLink, const RootedNetworkNode *targetNode, Node *unode) {
	assert(unode);
	assert(targetNode->isReticulation);
	// we only need to set link2 to the calling second parent...
	Link *link2 = unode->link->next;
	assert(link2->edge == nullptr);
	assert(link2->outer == nullptr);
	link2->edge = myLink->edge;
	link2->outer = myLink;

	// set the link directions
	unode->link->direction = Direction::INCOMING; // link to the first parent
	unode->link->next->direction = Direction::INCOMING; // link to the second parent
	unode->link->next->next->direction = Direction::OUTGOING; // link to the child
	unode->link->outer->direction = Direction::OUTGOING; // link from the first parent
	unode->link->next->outer->direction = Direction::OUTGOING; // link from the second parent
	unode->link->next->next->outer->direction = Direction::INCOMING; // link from the child

	return unode->link->next; // return the link2 that goes to the second encountered parent
}

Link* buildBackLinkReticulation(Link *myLink, const RootedNetworkNode *targetNode, const RootedNetworkNode *parentNode,
		size_t *inner_clv_index, size_t *inner_scaler_index, size_t *inner_pmatrix_index, size_t *inner_node_index, size_t *actNodeCount,
		size_t *actEdgeCount, size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	if (visitedReticulations.find(targetNode) == visitedReticulations.end()) { // first visit
		myLink->edge->length = targetNode->firstParentLength;
		return buildBackLinkReticulationFirstVisit(myLink, targetNode, parentNode, inner_clv_index, inner_scaler_index, inner_pmatrix_index,
				inner_node_index, actNodeCount, actEdgeCount, actLinkCount, network, visitedReticulations);
	} else {
		myLink->edge->length = targetNode->secondParentLength;
		return buildBackLinkReticulationSecondVisit(myLink, targetNode, visitedReticulations[targetNode]);
	}
}

Link* buildBackLinkInnerTree(Link *myLink, const RootedNetworkNode *targetNode, size_t *inner_clv_index, size_t *inner_scaler_index,
		size_t *inner_pmatrix_index, size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount, size_t *actLinkCount,
		Network &network, std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	Node *unode = &network.nodes[(*actNodeCount)++];
	Link *firstLink = &network.links[(*actLinkCount)++];
	Link *secondLink = &network.links[(*actLinkCount)++];
	Link *thirdLink = &network.links[(*actLinkCount)++];
	firstLink->next = secondLink;
	secondLink->next = thirdLink;
	thirdLink->next = firstLink;
	firstLink->node = unode;
	secondLink->node = unode;
	thirdLink->node = unode;

	Edge *firstEdge = myLink->edge;
	Edge *secondEdge = &network.edges[(*actEdgeCount)++];
	Edge *thirdEdge = &network.edges[(*actEdgeCount)++];
	//firstEdge->link1 is already set
	assert(firstEdge->link1 == myLink);
	secondEdge->link1 = secondLink;
	thirdEdge->link1 = thirdLink;

	// firstEdge->length is already set
	secondEdge->length = targetNode->children[0]->length;
	secondEdge->support = targetNode->children[0]->support;
	thirdEdge->length = targetNode->children[1]->length;
	thirdEdge->support = targetNode->children[1]->support;

	firstLink->edge = firstEdge;
	secondLink->edge = secondEdge;
	thirdLink->edge = thirdEdge;

	Link *link1Back = myLink;
	Link *link2Back = buildBackLink(secondLink, targetNode->children[0], targetNode, inner_clv_index, inner_scaler_index,
			inner_pmatrix_index, inner_node_index, actNodeCount, actEdgeCount, actLinkCount, network, visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, targetNode->children[1], targetNode, inner_clv_index, inner_scaler_index,
			inner_pmatrix_index, inner_node_index, actNodeCount, actEdgeCount, actLinkCount, network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	// firstEdge->link2 will be set by the caller
	secondEdge->link2 = link2Back;
	thirdEdge->link2 = link3Back;

	unode->initBasic((*inner_clv_index)++, (*inner_scaler_index)++, firstLink, targetNode->label);

	// set the indices now
	firstLink->node_index = (*inner_node_index)++;
	secondLink->node_index = (*inner_node_index)++;
	thirdLink->node_index = (*inner_node_index)++;
	// firstEdge->pmatrix_index is already set
	setPMatrixIndexConditional(secondEdge, inner_pmatrix_index);
	setPMatrixIndexConditional(thirdEdge, inner_pmatrix_index);

	network.inner_nodes.push_back(unode);

	assert(firstEdge->length != 0);
	assert(secondEdge->length != 0);
	assert(thirdEdge->length != 0);

	return firstLink;
}

Link* buildBackLinkLeaf(Link *myLink, const RootedNetworkNode *targetNode, size_t *actNodeCount, size_t *actLinkCount, Network &network) {
	Node *unode = &network.nodes[(*actNodeCount)++];
	Link *firstLink = &network.links[(*actLinkCount)++];
	firstLink->next = nullptr;
	firstLink->node = unode;

	Edge *firstEdge = myLink->edge;
	//firstEdge->link1 is already set, firstEdge->link2 will be set by the caller, firstEdge->length is already set

	firstLink->edge = firstEdge;
	Link *link1Back = myLink;
	firstLink->outer = link1Back;

	// set the indices now
	firstLink->node_index = targetNode->tip_index;
	// firstEdge->pmatrix_index is already set
	assert(targetNode->tip_index != std::numeric_limits<size_t>::max());
	unode->initBasic(targetNode->tip_index, -1, firstLink, targetNode->label);

	network.tip_nodes.push_back(unode);

	assert(firstEdge->length != 0);
	return firstLink;
}

Link* buildBackLink(Link *myLink, const RootedNetworkNode *targetNode, const RootedNetworkNode *parentNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index, size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network, std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	if (targetNode->isReticulation) {
		assert(targetNode->children.size() == 1);
		return buildBackLinkReticulation(myLink, targetNode, parentNode, inner_clv_index, inner_scaler_index, inner_pmatrix_index,
				inner_node_index, actNodeCount, actEdgeCount, actLinkCount, network, visitedReticulations);
	} else {
		if (targetNode->children.empty()) { // leaf node
			return buildBackLinkLeaf(myLink, targetNode, actNodeCount, actLinkCount, network);
		} else { // inner tree node
			assert(targetNode->children.size() == 2);
			return buildBackLinkInnerTree(myLink, targetNode, inner_clv_index, inner_scaler_index, inner_pmatrix_index, inner_node_index,
					actNodeCount, actEdgeCount, actLinkCount, network, visitedReticulations);
		}
	}
}

Network convertNetworkToplevelTrifurcation(const RootedNetwork &rnetwork, size_t node_count, size_t branch_count, RootedNetworkNode *root) {
	std::unordered_map<const RootedNetworkNode*, Node*> visitedReticulations;
	assert(root->children.size() == 3);

	size_t tip_count = rnetwork.tipCount;
	size_t link_count = branch_count * 2;

	Network network;
	network.nodes.resize(node_count);
	network.edges.resize(branch_count);
	network.links.resize(link_count);

	size_t actNodeCount = 0;
	size_t actEdgeCount = 0;
	size_t actLinkCount = 0;

	assert(root);

	size_t inner_clv_index = tip_count;
	size_t inner_scaler_index = 0;
	size_t inner_pmatrix_index = tip_count;
	size_t inner_node_index = tip_count;

	Node *uroot = &network.nodes[actNodeCount++];
	Link *firstLink = &network.links[actLinkCount++];
	Link *secondLink = &network.links[actLinkCount++];
	Link *thirdLink = &network.links[actLinkCount++];
	firstLink->next = secondLink;
	secondLink->next = thirdLink;
	thirdLink->next = firstLink;
	firstLink->node = uroot;
	secondLink->node = uroot;
	thirdLink->node = uroot;

	Edge *firstEdge = &network.edges[actEdgeCount++];
	Edge *secondEdge = &network.edges[actEdgeCount++];
	Edge *thirdEdge = &network.edges[actEdgeCount++];
	firstEdge->link1 = firstLink;
	secondEdge->link1 = secondLink;
	thirdEdge->link1 = thirdLink;

	firstEdge->length = root->children[0]->length;
	firstEdge->support = root->children[0]->support;
	secondEdge->length = root->children[1]->length;
	secondEdge->support = root->children[1]->support;
	thirdEdge->length = root->children[2]->length;
	thirdEdge->support = root->children[2]->support;

	firstLink->edge = firstEdge;
	secondLink->edge = secondEdge;
	thirdLink->edge = thirdEdge;

	Link *link1Back = buildBackLink(firstLink, root->children[0], root, &inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount, network, visitedReticulations);
	Link *link2Back = buildBackLink(secondLink, root->children[1], root, &inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount, network, visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, root->children[2], root, &inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount, network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	firstEdge->link2 = link1Back;
	secondEdge->link2 = link2Back;
	thirdEdge->link2 = link3Back;

	uroot->initBasic(inner_clv_index++, inner_scaler_index++, firstLink, root->label);

// set the indices now
	firstLink->node_index = inner_node_index++;
	secondLink->node_index = inner_node_index++;
	thirdLink->node_index = inner_node_index++;
	setPMatrixIndexConditional(firstEdge, &inner_pmatrix_index);
	setPMatrixIndexConditional(secondEdge, &inner_pmatrix_index);
	setPMatrixIndexConditional(thirdEdge, &inner_pmatrix_index);

	network.root = uroot;
	return network;
}

Network convertNetworkToplevelBifurcation(const RootedNetwork &rnetwork, size_t node_count, size_t branch_count, RootedNetworkNode *root) {
	std::unordered_map<const RootedNetworkNode*, Node*> visitedReticulations;
	assert(root->children.size() == 2);

	size_t tip_count = rnetwork.tipCount;
	size_t link_count = branch_count * 2;

	Network network;
	network.nodes.resize(node_count);
	network.edges.resize(branch_count);
	network.links.resize(link_count);

	size_t actNodeCount = 0;
	size_t actEdgeCount = 0;
	size_t actLinkCount = 0;

// create the uroot node.
	/* get the first root child that has descendants and make it the new root */
	RootedNetworkNode *new_root = nullptr;
	RootedNetworkNode *other_child = nullptr;
	if (root->children[0]->children.size() == 2) { // we don't want to have a reticulation node as the new root...
		new_root = root->children[0];
		other_child = root->children[1];
	} else {
		new_root = root->children[1];
		other_child = root->children[0];
	}
// the new 3 neighbors of new_root are now: new_root->children[0], new_root->children[1], and other_child
	assert(new_root);
	assert(other_child);

	size_t inner_clv_index = tip_count;
	size_t inner_scaler_index = 0;
	size_t inner_pmatrix_index = tip_count;
	size_t inner_node_index = tip_count;

	Node *uroot = &network.nodes[actNodeCount++];
	Link *firstLink = &network.links[actLinkCount++];
	Link *secondLink = &network.links[actLinkCount++];
	Link *thirdLink = &network.links[actLinkCount++];
	firstLink->next = secondLink;
	secondLink->next = thirdLink;
	thirdLink->next = firstLink;
	firstLink->node = uroot;
	secondLink->node = uroot;
	thirdLink->node = uroot;

	Edge *firstEdge = &network.edges[actEdgeCount++];
	Edge *secondEdge = &network.edges[actEdgeCount++];
	Edge *thirdEdge = &network.edges[actEdgeCount++];
	firstEdge->link1 = firstLink;
	secondEdge->link1 = secondLink;
	thirdEdge->link1 = thirdLink;

	firstEdge->length = new_root->children[0]->length;
	firstEdge->support = new_root->children[0]->support;
	secondEdge->length = new_root->children[1]->length;
	secondEdge->support = new_root->children[1]->support;
	thirdEdge->length = other_child->length + new_root->length;
	thirdEdge->support = std::min(other_child->support, new_root->support);

	firstLink->edge = firstEdge;
	secondLink->edge = secondEdge;
	thirdLink->edge = thirdEdge;

	Link *link1Back = buildBackLink(firstLink, new_root->children[0], new_root, &inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount, network, visitedReticulations);
	Link *link2Back = buildBackLink(secondLink, new_root->children[1], new_root, &inner_clv_index, &inner_scaler_index,
			&inner_pmatrix_index, &inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount, network, visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, other_child, root, &inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount, network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	firstEdge->link2 = link1Back;
	secondEdge->link2 = link2Back;
	thirdEdge->link2 = link3Back;

	uroot->initBasic(inner_clv_index++, inner_scaler_index++, firstLink, new_root->label);

// set the indices now
	firstLink->node_index = inner_node_index++;
	secondLink->node_index = inner_node_index++;
	thirdLink->node_index = inner_node_index++;
	setPMatrixIndexConditional(firstEdge, &inner_pmatrix_index);
	setPMatrixIndexConditional(secondEdge, &inner_pmatrix_index);
	setPMatrixIndexConditional(thirdEdge, &inner_pmatrix_index);

	network.root = uroot;
	return network;
}

Network convertNetwork(const RootedNetwork &rnetwork) {
	std::cout << exportDebugInfo(rnetwork) << "\n";
	assert(rnetwork.root->children.size() == 2 || rnetwork.root->children.size() == 3);

	size_t node_count = rnetwork.nodes.size();
// special case: check if rnetwork.root has only one child... if so, reset the root to its child.
	size_t branch_count = rnetwork.branchCount;
	RootedNetworkNode *root = rnetwork.root;
	if (root->children.size() == 1) {
		root = root->children[0];
		node_count--;
		branch_count--;
	}
	// now, the root has either 2 children (top-level bifurcation), or 3 children (top-level trifurcation).
	Network network;
	if (root->children.size() == 2) {
		network = convertNetworkToplevelBifurcation(rnetwork, node_count - 1, branch_count - 1, root);
	} else if (root->children.size() == 3) {
		network = convertNetworkToplevelTrifurcation(rnetwork, node_count, branch_count, root);
	} else {
		throw std::runtime_error("The network is not bifurcating");
	}

	// BUG: The sorting invalidates all the pointers!!! Disabled it for now.
//At the end, sort the arrays based on clv_index, pmatrix_index, node_index, reticulation_index...
	/*std::sort(network.nodes.begin(), network.nodes.end(),
	 [](const auto &lhs, const auto &rhs) {
	 return lhs.getClvIndex() < rhs.getClvIndex();
	 });
	 std::sort(network.edges.begin(), network.edges.end(),
	 [](const auto &lhs, const auto &rhs) {
	 return lhs.getPMatrixIndex() < rhs.getPMatrixIndex();
	 });
	 std::sort(network.links.begin(), network.links.end(),
	 [](const auto &lhs, const auto &rhs) {
	 return lhs.getNodeIndex() < rhs.getNodeIndex();
	 });
	 std::sort(network.reticulation_nodes.begin(),
	 network.reticulation_nodes.end(),
	 [](const auto &lhs, const auto &rhs) {
	 return lhs->getReticulationData()->getReticulationIndex()
	 < rhs->getReticulationData()->getReticulationIndex();
	 });*/

	assert(!network.root->isTip());

	// ensure that no branch lengths are zero
	for (size_t i = 0; i < network.edges.size(); ++i) {
		assert(network.edges[i].length != 0);
	}

	return network;
}

Network readNetworkFromString(const std::string &newick) {
	RootedNetwork *rnetwork = parseRootedNetworkFromNewickString(newick);
	Network network = convertNetwork(*rnetwork);
	delete rnetwork;
	return network;
}
Network readNetworkFromFile(const std::string &filename) {
	std::ifstream t(filename);
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string newick = buffer.str();
	return readNetworkFromString(newick);
}

std::string newickNodeName(const Node* node, const Node* parent) {
	assert(node);
	std::stringstream sb("");

	sb << node->label;
	if (node->getType() == NodeType::RETICULATION_NODE) {
		assert(parent);
		sb << "#" << node->getReticulationData()->getLabel();
		Link* link = node->getReticulationData()->getLinkToFirstParent();
		double prob = node->getReticulationData()->getProb(0);
		if (node->getReticulationData()->getLinkToSecondParent()->getTargetNode() == parent) {
			link = node->getReticulationData()->getLinkToSecondParent();
			prob = 1.0 - prob;
		} else {
			assert(node->getReticulationData()->getLinkToFirstParent()->getTargetNode() == parent);
		}

		sb << ":" << link->edge->length << ":";
		if (link->edge->support != 0.0) {
			sb << link->edge->support;
		}
		sb << ":" << prob;
	} else {
		if (parent != nullptr) {
			sb << ":" << node->getEdgeTo(parent)->length;
			if (node->getEdgeTo(parent)->support != 0.0) {
				sb << ":" << node->getEdgeTo(parent)->support;
			}
		}
	}
	return sb.str();
}

std::string printNodeNewick(const Node* node, const Node* parent, std::unordered_set<const Node*>& visited_reticulations) {
	std::stringstream sb("");
	std::vector<Node*> children = node->getChildren(parent);
	if (!children.empty() && visited_reticulations.find(node) == visited_reticulations.end()) {
		sb << "(";
		for (size_t i = 0; i < children.size() - 1; i++) {
			sb << printNodeNewick(children[i], node, visited_reticulations);
			sb << ",";
		}
		sb << printNodeNewick(children[children.size() - 1], node, visited_reticulations);
		sb << ")";
		if (node->getType() == NodeType::RETICULATION_NODE) {
			visited_reticulations.insert(node);
		}
	}
	sb << newickNodeName(node, parent);
	return sb.str();
}

std::string toExtendedNewick(const Network &network) {
	std::unordered_set<const Node*> visited_reticulations;
	return printNodeNewick(network.root, nullptr, visited_reticulations) + ";";
}

}
