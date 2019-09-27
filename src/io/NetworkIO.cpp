/*
 * NetworkConverter.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "NetworkIO.hpp"
#include "../Network.hpp"

#include <array>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <string>
#include <sstream>

extern "C" {
#include "lowlevel_parsing.hpp"
}

namespace netrax {

void setLinksAndEdgesAndNodes(Network &network, const unetwork_node_t *unode,
		bool keepDirection) {
	if (!unode->next) { // leaf node
		assert(!keepDirection);
		network.links[unode->node_index].init(unode->node_index,
				&network.nodes[unode->clv_index],
				&network.edges[unode->pmatrix_index], nullptr,
				&network.links[unode->back->node_index], Direction::UNDEFINED);
		network.edges[unode->pmatrix_index].init(unode->pmatrix_index,
				&network.links[unode->node_index],
				&network.links[unode->back->node_index], unode->length);
		network.tip_nodes.push_back(&network.nodes[unode->clv_index]);
	} else { // inner node
		Direction dir1, dir2, dir3 = Direction::UNDEFINED;
		if (keepDirection) {
			dir1 = unode->incoming ? Direction::INCOMING : Direction::OUTGOING;
			dir2 = unode->next->incoming ?
					Direction::INCOMING : Direction::OUTGOING;
			dir3 = unode->next->next->incoming ?
					Direction::INCOMING : Direction::OUTGOING;
		}
		network.links[unode->node_index].init(unode->node_index,
				&network.nodes[unode->clv_index],
				&network.edges[unode->pmatrix_index],
				&network.links[unode->next->node_index],
				&network.links[unode->back->node_index], dir1);
		network.links[unode->next->node_index].init(unode->next->node_index,
				&network.nodes[unode->clv_index],
				&network.edges[unode->next->pmatrix_index],
				&network.links[unode->next->next->node_index],
				&network.links[unode->next->back->node_index], dir2);
		network.links[unode->next->next->node_index].init(
				unode->next->next->node_index, &network.nodes[unode->clv_index],
				&network.edges[unode->next->next->pmatrix_index],
				&network.links[unode->node_index],
				&network.links[unode->next->next->back->node_index], dir3);

		network.edges[unode->pmatrix_index].init(unode->pmatrix_index,
				&network.links[unode->node_index],
				&network.links[unode->back->node_index], unode->length);
		network.edges[unode->next->pmatrix_index].init(
				unode->next->pmatrix_index,
				&network.links[unode->next->node_index],
				&network.links[unode->next->back->node_index],
				unode->next->length);
		network.edges[unode->next->next->pmatrix_index].init(
				unode->next->next->pmatrix_index,
				&network.links[unode->next->next->node_index],
				&network.links[unode->next->next->back->node_index],
				unode->next->next->length);

		network.inner_nodes.push_back(&network.nodes[unode->clv_index]);
	}
}

Network convertNetwork(const unetwork_t &unetwork) {
	Network network;
	network.nodes.resize(
			unetwork.inner_tree_count + unetwork.tip_count
					+ unetwork.reticulation_count);
	network.edges.resize(unetwork.edge_count);
	network.links.resize(
			unetwork.tip_count + unetwork.reticulation_count * 3
					+ unetwork.inner_tree_count * 3);
	network.reticulation_nodes.resize(unetwork.reticulation_count);

	for (size_t i = 0; i < network.nodes.size(); ++i) {
		unetwork_node_t *unode = unetwork.nodes[i];

		if (unode->reticulation_index == -1) { // basic node
			network.nodes[unode->clv_index].initBasic(unode->clv_index,
					unode->scaler_index, &network.links[unode->node_index],
					unode->label ? unode->label : "");
			setLinksAndEdgesAndNodes(network, unode, false);
		} else { // reticulation node
			ReticulationData retData;

			unetwork_node_t *first_parent_unode;
			unetwork_node_t *second_parent_unode;
			unetwork_node_t *child_unode;
			if (unode->incoming) {
				first_parent_unode = unode;
				if (unode->next->incoming) {
					second_parent_unode = unode->next;
					child_unode = unode->next->next;
				} else {
					child_unode = unode->next;
					second_parent_unode = unode->next->next;
				}
			} else {
				child_unode = unode;
				first_parent_unode = unode->next;
				second_parent_unode = unode->next->next;
			}
			assert(first_parent_unode->incoming);
			assert(second_parent_unode->incoming);
			assert(!child_unode->incoming);

			retData.init(first_parent_unode->reticulation_index,
					first_parent_unode->reticulation_name ?
							first_parent_unode->reticulation_name : "", 0,
					&network.links[first_parent_unode->node_index],
					&network.links[second_parent_unode->node_index],
					&network.links[child_unode->node_index],
					first_parent_unode->prob);
			network.nodes[first_parent_unode->clv_index].initReticulation(
					first_parent_unode->clv_index,
					first_parent_unode->scaler_index,
					&network.links[first_parent_unode->node_index],
					first_parent_unode->label ? first_parent_unode->label : "",
					retData);

			setLinksAndEdgesAndNodes(network, unode, true);
			network.reticulation_nodes[unode->reticulation_index] =
					&network.nodes[unode->clv_index];
		}
	}
	network.root = &network.nodes[unetwork.vroot->clv_index];

	assert(network.tip_nodes.size() == unetwork.tip_count);
	assert(
			network.inner_nodes.size()
					== unetwork.inner_tree_count + unetwork.reticulation_count);

	return network;
}

Link* buildBackLink(Link *myLink, const RootedNetworkNode *targetNode,
		const RootedNetworkNode *parentNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index,
		size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations);

Link* buildBackLinkReticulationFirstVisit(Link *myLink,
		const RootedNetworkNode *targetNode,
		const RootedNetworkNode *parentNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index,
		size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	assert(targetNode->isReticulation);

	const RootedNetworkNode *firstEncounteredParent;
	const RootedNetworkNode *secondEncounteredParent;
	double firstEncounteredParentLength, secondEncounteredParentLength,
			firstEncounteredParentSupport, secondEncounteredParentSupport,
			firstEncounteredParentProb, secondEncounteredParentProb;
	if (targetNode->firstParent == parentNode) {
		firstEncounteredParent = targetNode->firstParent;
		firstEncounteredParentLength = targetNode->firstParentLength;
		firstEncounteredParentSupport = targetNode->firstParentSupport;
		firstEncounteredParentProb = targetNode->firstParentProb;
		secondEncounteredParent = targetNode->secondParent;
		secondEncounteredParentLength = targetNode->secondParentLength;
		secondEncounteredParentSupport = targetNode->secondParentSupport;
		secondEncounteredParentProb = targetNode->secondParentProb;
	} else {
		assert(targetNode->secondParent == parentNode);
		firstEncounteredParent = targetNode->secondParent;
		secondEncounteredParent = targetNode->firstParent;
		firstEncounteredParentLength = targetNode->secondParentLength;
		firstEncounteredParentSupport = targetNode->secondParentSupport;
		firstEncounteredParentProb = targetNode->secondParentProb;
		secondEncounteredParentLength = targetNode->firstParentLength;
		secondEncounteredParentSupport = targetNode->firstParentSupport;
		secondEncounteredParentProb = targetNode->firstParentProb;
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
	//Edge *secondEdge = &network.edges[(*actEdgeCount)++];
	Edge *thirdEdge = &network.edges[(*actEdgeCount)++];
	//firstEdge->link1 is already set
	assert(firstEdge->link1 == myLink);
	thirdEdge->link1 = thirdLink;

	// firstEdge->length is already set
	assert(firstEdge->length == targetNode->firstParentLength);
	thirdEdge->length = targetNode->children[0]->length;

	firstLink->edge = firstEdge;
	secondLink->edge = nullptr; // will be set later
	thirdLink->edge = thirdEdge;

	Link *link1Back = myLink;
	Link *link2Back = nullptr; // will be set later
	Link *link3Back = buildBackLink(thirdLink, targetNode->children[0],
			targetNode, inner_clv_index, inner_scaler_index,
			inner_pmatrix_index, inner_node_index, actNodeCount, actEdgeCount,
			actLinkCount, network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	// firstEdge->link2 will be set by the caller
	thirdEdge->link2 = link3Back;

	// set the indices now
	firstLink->node_index = (*inner_node_index)++;
	secondLink->node_index = (*inner_node_index)++;
	thirdLink->node_index = (*inner_node_index)++;
	// firstEdge->pmatrix_index is already set
	thirdEdge->pmatrix_index = (*inner_pmatrix_index)++;

	ReticulationData retData;
	retData.init(targetNode->reticulationId, targetNode->reticulationName, 0,
			firstLink, secondLink, thirdLink, firstEncounteredParentProb);
	unode->initReticulation((*inner_clv_index)++, (*inner_scaler_index)++,
			firstLink, targetNode->label, retData);

	network.inner_nodes.push_back(unode);
	network.reticulation_nodes.push_back(unode);
	visitedReticulations[targetNode] = unode;

	return firstLink;
}

Link* buildBackLinkReticulationSecondVisit(Link *myLink,
		const RootedNetworkNode *targetNode, Network &network, Node *unode) {
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

Link* buildBackLinkReticulation(Link *myLink,
		const RootedNetworkNode *targetNode,
		const RootedNetworkNode *parentNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index,
		size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	if (visitedReticulations.find(targetNode) == visitedReticulations.end()) { // first visit
		return buildBackLinkReticulationFirstVisit(myLink, targetNode,
				parentNode, inner_clv_index, inner_scaler_index,
				inner_pmatrix_index, inner_node_index, actNodeCount,
				actEdgeCount, actLinkCount, network, visitedReticulations);
	} else {
		return buildBackLinkReticulationSecondVisit(myLink, targetNode, network,
				visitedReticulations[targetNode]);
	}
}

Link* buildBackLinkInnerTree(Link *myLink, const RootedNetworkNode *targetNode,
		size_t *inner_clv_index, size_t *inner_scaler_index,
		size_t *inner_pmatrix_index, size_t *inner_node_index,
		size_t *actNodeCount, size_t *actEdgeCount, size_t *actLinkCount,
		Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
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
	thirdEdge->length = targetNode->children[1]->length;

	firstLink->edge = firstEdge;
	secondLink->edge = secondEdge;
	thirdLink->edge = thirdEdge;

	Link *link1Back = myLink;
	Link *link2Back = buildBackLink(secondLink, targetNode->children[0],
			targetNode, inner_clv_index, inner_scaler_index,
			inner_pmatrix_index, inner_node_index, actNodeCount, actEdgeCount,
			actLinkCount, network, visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, targetNode->children[1],
			targetNode, inner_clv_index, inner_scaler_index,
			inner_pmatrix_index, inner_node_index, actNodeCount, actEdgeCount,
			actLinkCount, network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	// firstEdge->link2 will be set by the caller
	secondEdge->link2 = link2Back;
	thirdEdge->link2 = link3Back;

	// set the indices now
	firstLink->node_index = (*inner_node_index)++;
	secondLink->node_index = (*inner_node_index)++;
	thirdLink->node_index = (*inner_node_index)++;
	// firstEdge->pmatrix_index is already set
	secondEdge->pmatrix_index = (*inner_pmatrix_index)++;
	thirdEdge->pmatrix_index = (*inner_pmatrix_index)++;
	unode->initBasic((*inner_clv_index)++, (*inner_scaler_index)++, firstLink,
			targetNode->label);

	network.inner_nodes.push_back(unode);

	return firstLink;
}

Link* buildBackLinkLeaf(Link *myLink, const RootedNetworkNode *targetNode,
		size_t *actNodeCount, size_t *actLinkCount, Network &network) {
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
	return firstLink;
}

Link* buildBackLink(Link *myLink, const RootedNetworkNode *targetNode,
		const RootedNetworkNode *parentNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index,
		size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	if (targetNode->isReticulation) {
		assert(targetNode->children.size() == 1);
		return buildBackLinkReticulation(myLink, targetNode, parentNode,
				inner_clv_index, inner_scaler_index, inner_pmatrix_index,
				inner_node_index, actNodeCount, actEdgeCount, actLinkCount,
				network, visitedReticulations);
	} else {
		if (targetNode->children.empty()) { // leaf node
			return buildBackLinkLeaf(myLink, targetNode, actNodeCount,
					actLinkCount, network);
		} else { // inner tree node
			assert(targetNode->children.size() == 2);
			return buildBackLinkInnerTree(myLink, targetNode, inner_clv_index,
					inner_scaler_index, inner_pmatrix_index, inner_node_index,
					actNodeCount, actEdgeCount, actLinkCount, network,
					visitedReticulations);
		}
	}
}

Network convertNetworkToplevelTrifurcation(const RootedNetwork &rnetwork,
		size_t node_count, RootedNetworkNode *root) {
	std::unordered_map<const RootedNetworkNode*, Node*> visitedReticulations;
	assert(root->children.size() == 3);

	size_t tip_count = rnetwork.tipCount;
	size_t inner_count = node_count - tip_count;
	size_t branch_count = inner_count * 3; // assuming binary network
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
	secondEdge->length = root->children[1]->length;
	thirdEdge->length = root->children[2]->length;

	firstLink->edge = firstEdge;
	secondLink->edge = secondEdge;
	thirdLink->edge = thirdEdge;

	Link *link1Back = buildBackLink(firstLink, root->children[0], root,
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);
	Link *link2Back = buildBackLink(secondLink, root->children[1], root,
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, root->children[2], root,
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	firstEdge->link2 = link1Back;
	secondEdge->link2 = link2Back;
	thirdEdge->link2 = link3Back;

// set the indices now
	firstLink->node_index = inner_node_index++;
	secondLink->node_index = inner_node_index++;
	thirdLink->node_index = inner_node_index++;
	firstEdge->pmatrix_index = inner_pmatrix_index++;
	secondEdge->pmatrix_index = inner_pmatrix_index++;
	thirdEdge->pmatrix_index = inner_pmatrix_index++;
	uroot->initBasic(inner_clv_index++, inner_scaler_index++, thirdLink,
			root->label);
	network.root = uroot;
	return network;
}

Network convertNetworkToplevelBifurcation(const RootedNetwork &rnetwork,
		size_t node_count, RootedNetworkNode *root) {
	std::unordered_map<const RootedNetworkNode*, Node*> visitedReticulations;
	assert(root->children.size() == 2);

	size_t tip_count = rnetwork.tipCount;
	size_t inner_count = node_count - tip_count;
	size_t branch_count = inner_count * 3; // assuming binary network
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
	secondEdge->length = new_root->children[1]->length;
	thirdEdge->length = other_child->length;

	firstLink->edge = firstEdge;
	secondLink->edge = secondEdge;
	thirdLink->edge = thirdEdge;

	Link *link1Back = buildBackLink(firstLink, new_root->children[0], new_root,
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);
	Link *link2Back = buildBackLink(secondLink, new_root->children[1], new_root,
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, other_child, root,
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);

	firstLink->outer = link1Back;
	secondLink->outer = link2Back;
	thirdLink->outer = link3Back;
	firstEdge->link2 = link1Back;
	secondEdge->link2 = link2Back;
	thirdEdge->link2 = link3Back;

// set the indices now
	firstLink->node_index = inner_node_index++;
	secondLink->node_index = inner_node_index++;
	thirdLink->node_index = inner_node_index++;
	firstEdge->pmatrix_index = inner_pmatrix_index++;
	secondEdge->pmatrix_index = inner_pmatrix_index++;
	thirdEdge->pmatrix_index = inner_pmatrix_index++;
	uroot->initBasic(inner_clv_index++, inner_scaler_index++, thirdLink,
			new_root->label);
	network.root = uroot;
	return network;
}

Network convertNetwork(const RootedNetwork &rnetwork) {
	size_t node_count = rnetwork.nodes.size();
// special case: check if rnetwork.root has only one child... if so, reset the root to its child.
	RootedNetworkNode *root = rnetwork.root;
	if (root->children.size() == 1) {
		root = root->children[0];
		node_count--;
	}
	// now, the root has either 2 children (top-level bifurcation), or 3 children (top-level trifurcation).
	Network network;
	if (root->children.size() == 2) {
		network = convertNetworkToplevelBifurcation(rnetwork, node_count, root);
	} else if (root->children.size() == 3) {
		network = convertNetworkToplevelTrifurcation(rnetwork, node_count, root);
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

	return network;
}

Network readNetworkFromString(const std::string &newick) {
	//unetwork_t *unetwork = unetwork_parse_newick_string(newick.c_str());
	//return convertNetwork(*unetwork);
	RootedNetwork rnetwork = parseRootedNetworkFromNewickString(newick);
	return convertNetwork(rnetwork);
}
Network readNetworkFromFile(const std::string &filename) {
	//unetwork_t *unetwork = unetwork_parse_newick(filename.c_str());
	//return convertNetwork(*unetwork);
	std::ifstream t(filename);
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string newick = buffer.str();
	return readNetworkFromString(newick);
}
std::string toExtendedNewick(const Network &network) {
	throw std::runtime_error("This function has not been implemented yet");
}

}
