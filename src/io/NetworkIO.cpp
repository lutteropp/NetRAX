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
		size_t *inner_clv_index, size_t *inner_scaler_index,
		size_t *inner_pmatrix_index, size_t *inner_node_index,
		size_t *actNodeCount, size_t *actEdgeCount, size_t *actLinkCount,
		Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations);

Link* buildBackLinkReticulationFirstVisit(Link *myLink,
		const RootedNetworkNode *targetNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index,
		size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	// we will keep link2 for the second parent...
	// TODO: Remember setting the link directions here
}

Link* buildBackLinkReticulationSecondVisit(Link *myLink,
		const RootedNetworkNode *targetNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index,
		size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network, Node *unode) {
	// we only need to set link2...
	// TODO: Remember setting the link directions here
}

Link* buildBackLinkReticulation(Link *myLink,
		const RootedNetworkNode *targetNode, size_t *inner_clv_index,
		size_t *inner_scaler_index, size_t *inner_pmatrix_index,
		size_t *inner_node_index, size_t *actNodeCount, size_t *actEdgeCount,
		size_t *actLinkCount, Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	bool firstVisit = true;
	Node *unode = nullptr;
	if (visitedReticulations.find(targetNode) != visitedReticulations.end()) {
		return buildBackLinkReticulationFirstVisit(myLink, targetNode,
				inner_clv_index, inner_scaler_index, inner_pmatrix_index,
				inner_node_index, actNodeCount, actEdgeCount, actLinkCount,
				network, visitedReticulations);
	} else {
		return buildBackLinkReticulationSecondVisit(myLink, targetNode,
				inner_clv_index, inner_scaler_index, inner_pmatrix_index,
				inner_node_index, actNodeCount, actEdgeCount, actLinkCount,
				network, visitedReticulations[targetNode]);
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
			inner_clv_index, inner_scaler_index, inner_pmatrix_index,
			inner_node_index, actNodeCount, actEdgeCount, actLinkCount, network,
			visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, targetNode->children[0],
			inner_clv_index, inner_scaler_index, inner_pmatrix_index,
			inner_node_index, actNodeCount, actEdgeCount, actLinkCount, network,
			visitedReticulations);

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
		size_t *inner_clv_index, size_t *inner_scaler_index,
		size_t *inner_pmatrix_index, size_t *inner_node_index,
		size_t *actNodeCount, size_t *actEdgeCount, size_t *actLinkCount,
		Network &network) {
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
	firstLink->node_index = (*inner_node_index)++;
	// firstEdge->pmatrix_index is already set
	assert(targetNode->tip_index != std::numeric_limits<size_t>::infinity());
	unode->initBasic(targetNode->tip_index, -1, firstLink, targetNode->label);

	network.tip_nodes.push_back(unode);
	return firstLink;
}

Link* buildBackLink(Link *myLink, const RootedNetworkNode *targetNode,
		size_t *inner_clv_index, size_t *inner_scaler_index,
		size_t *inner_pmatrix_index, size_t *inner_node_index,
		size_t *actNodeCount, size_t *actEdgeCount, size_t *actLinkCount,
		Network &network,
		std::unordered_map<const RootedNetworkNode*, Node*> &visitedReticulations) {
	if (targetNode->isReticulation) {
		assert(targetNode->children.size() == 1);
	} else {
		if (targetNode->children.empty()) { // leaf node
			return buildBackLinkLeaf(myLink, targetNode, inner_clv_index,
					inner_scaler_index, inner_pmatrix_index, inner_node_index,
					actNodeCount, actEdgeCount, actLinkCount, network);
		} else { // inner tree node
			assert(targetNode->children.size() == 2);
			return buildBackLinkInnerTree(myLink, targetNode, inner_clv_index,
					inner_scaler_index, inner_pmatrix_index, inner_node_index,
					actNodeCount, actEdgeCount, actLinkCount, network,
					visitedReticulations);
		}
	}
}

Network convertNetwork(const RootedNetwork &rnetwork) {
	std::unordered_map<const RootedNetworkNode*, Node*> visitedReticulations;
	size_t node_count = rnetwork.nodes.size();
	// special case: check if rnetwork.root has only one child... if so, reset the root to its child.
	RootedNetworkNode *root = rnetwork.root;
	if (root->children.size() == 1) {
		root = root->children[0];
		node_count--;
	}
	assert(root->children.size() == 2);
	size_t reticulation_count = rnetwork.reticulationCount;
	size_t tip_count = rnetwork.tipCount;
	size_t inner_count = node_count - tip_count;
	size_t branch_count = inner_count * 3; // assuming binary network
	size_t link_count = branch_count * 2;

	Network network;
	network.nodes.resize(node_count);
	network.edges.resize(branch_count);
	network.links.resize(link_count);
	network.reticulation_nodes.resize(reticulation_count);

	size_t actNodeCount = 0;
	size_t actEdgeCount = 0;
	size_t actLinkCount = 0;
	size_t actReticulationCount = 0;

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

	Link *link1Back = buildBackLink(firstLink, new_root->children[0],
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);
	Link *link2Back = buildBackLink(secondLink, new_root->children[1],
			&inner_clv_index, &inner_scaler_index, &inner_pmatrix_index,
			&inner_node_index, &actNodeCount, &actEdgeCount, &actLinkCount,
			network, visitedReticulations);
	Link *link3Back = buildBackLink(thirdLink, other_child, &inner_clv_index,
			&inner_scaler_index, &inner_pmatrix_index, &inner_node_index,
			&actNodeCount, &actEdgeCount, &actLinkCount, network,
			visitedReticulations);

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

	//At the end, sort the arrays based on clv_index, pmatrix_index, node_index...
	std::sort(network.nodes.begin(), network.nodes.end(),
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

	return network;
}

Network readNetworkFromString(const std::string &newick) {
	unetwork_t *unetwork = unetwork_parse_newick_string(newick.c_str());
	return convertNetwork(*unetwork);
}
Network readNetworkFromFile(const std::string &filename) {
	unetwork_t *unetwork = unetwork_parse_newick(filename.c_str());
	return convertNetwork(*unetwork);
}
std::string toExtendedNewick(const Network &network) {
	throw std::runtime_error("This function has not been implemented yet");
}

}
