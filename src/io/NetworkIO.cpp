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
#include <stack>

#include "../graph/Direction.hpp"
#include "../graph/Edge.hpp"
#include "../graph/Link.hpp"
#include "../graph/Network.hpp"
#include "../graph/Node.hpp"
#include "../graph/NodeType.hpp"
#include "../graph/ReticulationData.hpp"

namespace netrax {


std::vector<RootedNetworkNode*> collectNodes(RootedNetwork& rnetwork) {
	std::vector<RootedNetworkNode*> res;
	std::stack<RootedNetworkNode*> s;
	std::unordered_set<RootedNetworkNode*> visited;
	s.emplace(rnetwork.root);
	while (!s.empty()) {
		RootedNetworkNode* actNode = s.top();
		s.pop();
		visited.emplace(actNode);
		res.emplace_back(actNode);
		for (RootedNetworkNode* child : actNode->children) {
			if (visited.find(child) == visited.end()) {
				s.emplace(child);
			}
		}
	}
	return res;
}

Link* make_link(size_t link_id, Node* node, Edge* edge, Direction dir) {
	Link link;
	link.init(link_id, node, edge, nullptr, nullptr, dir);
	return node->addLink(link);
}

Network convertNetworkToplevelTrifurcation(RootedNetwork& rnetwork, size_t node_count, size_t branch_count) {
	Network network;

	network.nodes.resize(node_count);

	network.edges.resize(branch_count);
	network.links.resize(2 * branch_count);
	network.reticulation_nodes.resize(rnetwork.reticulationCount);
	network.tip_nodes.resize(rnetwork.tipCount);
	network.inner_nodes.resize(node_count - rnetwork.tipCount);

	network.root = &network.nodes[network.nodes.size() - 1 - rnetwork.reticulationCount];

	std::vector<RootedNetworkNode*> rnetwork_nodes = collectNodes(rnetwork);
	std::vector<RootedNetworkNode*> rnetwork_tips, rnetwork_inner_tree, rnetwork_reticulations;
	for (RootedNetworkNode* ptr : rnetwork_nodes) {
		if (ptr->children.empty()) {
			rnetwork_tips.emplace_back(ptr);
		} else {
			if (ptr->isReticulation) {
				rnetwork_reticulations.emplace_back(ptr);
			} else {
				if (ptr != rnetwork.root) {
					rnetwork_inner_tree.emplace_back(ptr);
				}
			}
		}
	}
	rnetwork_inner_tree.emplace_back(rnetwork.root);

	size_t n_links = 0;

	// 1.) Create all the nodes and edges. Also create all the incoming links.
	for (RootedNetworkNode* rnode : rnetwork_tips) {
		size_t clv_index = rnode->tip_index;
		rnode->clv_index = clv_index;
		int scaler_index = -1;
		assert(clv_index < network.nodes.size());
		network.nodes[clv_index].initBasic(clv_index, scaler_index, rnode->label);
		size_t pmatrix_index = clv_index;
		assert(pmatrix_index < network.edges.size());
		network.edges[pmatrix_index].init(pmatrix_index, nullptr, nullptr, rnode->length);
		network.tip_nodes[clv_index] = &network.nodes[clv_index];

		Link* linkToParent = make_link(n_links, &network.nodes[clv_index], &network.edges[pmatrix_index], Direction::INCOMING);
		network.links[n_links] = linkToParent;
		network.edges[pmatrix_index].link1 = linkToParent;
		n_links++;
	}

	for (size_t i = 0; i < rnetwork_inner_tree.size(); ++i) {
		RootedNetworkNode* rnode = rnetwork_inner_tree[i];
		size_t clv_index = i + rnetwork_tips.size();
		rnode->clv_index = clv_index;
		int scaler_index = i;

		assert(clv_index < network.nodes.size());
		network.nodes[clv_index].initBasic(clv_index, scaler_index, rnode->label);

		if (rnode != rnetwork.root) {
			size_t pmatrix_index = clv_index;
			assert(pmatrix_index < network.edges.size());
			network.edges[pmatrix_index].init(pmatrix_index, nullptr, nullptr, rnode->length);
			Link* linkToParent = make_link(n_links, &network.nodes[clv_index], &network.edges[pmatrix_index], Direction::INCOMING);
			network.links[n_links] = linkToParent;
			network.edges[pmatrix_index].link1 = linkToParent;
			n_links++;
			assert(rnode->children.size() == 2);
		} else {
			assert(rnode->children.size() == 3);
		}

		network.inner_nodes[i] = &network.nodes[clv_index];
	}

	for (size_t i = 0; i < rnetwork_reticulations.size(); ++i) {
		RootedNetworkNode* rnode = rnetwork_reticulations[i];
		size_t clv_index = i + rnetwork_tips.size() + rnetwork_inner_tree.size();
		rnode->clv_index = clv_index;
		rnode->reticulation_index = i;
		int scaler_index = i + rnetwork_inner_tree.size();
		ReticulationData retData;
		retData.init(i, rnode->reticulationName, false, nullptr, nullptr, nullptr, rnode->firstParentProb);
		assert(clv_index < network.nodes.size());
		network.nodes[clv_index].initReticulation(clv_index, scaler_index, rnode->label, retData);

		size_t pmatrix_index = rnetwork_tips.size() + rnetwork_inner_tree.size() - 1 + 2 * i;
		assert(pmatrix_index + 1 < network.edges.size());
		network.edges[pmatrix_index].init(pmatrix_index, nullptr, nullptr, rnode->firstParentLength);
		network.edges[pmatrix_index + 1].init(pmatrix_index + 1, nullptr, nullptr, rnode->secondParentLength);
		network.reticulation_nodes[i] = &network.nodes[clv_index];
		network.inner_nodes[i + rnetwork_inner_tree.size()] = &network.nodes[clv_index];

		Link* linkToFirstParent = make_link(n_links, &network.nodes[clv_index], &network.edges[pmatrix_index], Direction::INCOMING);
		network.links[n_links] = linkToFirstParent;
		network.edges[pmatrix_index].link1 = linkToFirstParent;

		Link* linkToSecondParent = make_link(n_links + 1, &network.nodes[clv_index], &network.edges[pmatrix_index + 1], Direction::INCOMING);
		network.links[n_links + 1] = linkToSecondParent;
		network.edges[pmatrix_index + 1].link1 = linkToSecondParent;
		n_links += 2;
	}

	// 2.) Create all the outgoing links
	for (const auto& rnode : rnetwork_nodes) {
		if (rnode == rnetwork.root) {
			continue;
		}
		if (rnode->isReticulation) { // 2 parents
			size_t pmatrix_index = rnetwork_tips.size() + rnetwork_inner_tree.size() - 1 + 2 * rnode->reticulation_index;

			assert(rnode->firstParent->clv_index < network.nodes.size());
			assert(rnode->secondParent->clv_index < network.nodes.size());
			assert(pmatrix_index + 1 < network.edges.size());

			Link* linkFromFirstParent = make_link(n_links, &network.nodes[rnode->firstParent->clv_index], &network.edges[pmatrix_index], Direction::OUTGOING);
			network.links[n_links] = linkFromFirstParent;
			Link* linkFromSecondParent = make_link(n_links + 1, &network.nodes[rnode->secondParent->clv_index], &network.edges[pmatrix_index + 1], Direction::OUTGOING);
			network.links[n_links] = linkFromSecondParent;

			assert(pmatrix_index + 1 < network.edges.size());
			network.edges[pmatrix_index].link2 = linkFromFirstParent;
			network.edges[pmatrix_index + 1].link2 = linkFromSecondParent;

			n_links += 2;
		} else { // 1 parent
			size_t pmatrix_index = rnode->clv_index;

			assert(rnode->parent->clv_index < network.nodes.size());
			assert(pmatrix_index < network.edges.size());

			Link* linkFromParent = make_link(n_links, &network.nodes[rnode->parent->clv_index], &network.edges[pmatrix_index], Direction::OUTGOING);
			network.links[n_links] = linkFromParent;
			network.edges[pmatrix_index].link2 = linkFromParent;

			n_links++;
		}
	}

	// 3.) Create the outer links
	for (const auto& rnode : rnetwork_nodes) {
		if (rnode == rnetwork.root) {
			continue;
		}
		if (rnode->isReticulation) { // 2 parents
			size_t pmatrix_index = rnetwork_tips.size() + rnetwork_inner_tree.size() - 1 + 2 * rnode->reticulation_index;
			Link* linkFromFirstParent = network.edges[pmatrix_index].link2;
			Link* linkFromSecondParent = network.edges[pmatrix_index + 1].link2;

			Link* linkToFirstParent = network.nodes[rnode->clv_index].getLinkToClvIndex(rnode->firstParent->clv_index);
			Link* linkToSecondParent = network.nodes[rnode->clv_index].getLinkToClvIndex(rnode->secondParent->clv_index);

			linkFromFirstParent->outer = linkToFirstParent;
			linkToFirstParent->outer = linkFromFirstParent;
			linkFromSecondParent->outer = linkToSecondParent;
			linkToSecondParent->outer = linkFromSecondParent;
		} else { // 1 parent
			size_t pmatrix_index = rnode->clv_index;
			Link* linkFromParent = network.edges[pmatrix_index].link2;

			Link* linkToParent = network.nodes[rnode->clv_index].getLinkToClvIndex(rnode->parent->clv_index);
			linkFromParent->outer = linkToParent;
			linkToParent->outer = linkFromParent;
		}
	}

	// check that all links are sane
	for (size_t i = 0; i < network.links.size(); ++i) {
		assert(network.links[i]);
		assert(network.links[i]->outer);
		assert(network.links[i] != network.links[i]->outer);
	}

	return network;
}

bool networkIsConnected(const Network& network) {
	unsigned int n_visited;
	std::vector<bool> visited(network.num_nodes(), false);
	std::stack<const Node*> s;
	s.emplace(network.root);
	while (!s.empty()) {
		const Node* actNode = s.top();
		s.pop();
		visited[actNode->clv_index] = true;
		n_visited++;
		for (const Node* neigh : actNode->getNeighbors()) {
			if (!visited[neigh->clv_index]) {
				s.emplace(neigh);
			}
		}
	}
	return (n_visited == network.num_nodes());
}

std::pair<size_t, size_t> makeToplevelTrifurcation(RootedNetwork& rnetwork) {
	RootedNetworkNode *root = rnetwork.root;
	size_t node_count = rnetwork.nodes.size();
	size_t branch_count = rnetwork.branchCount;
	if (root->children.size() == 3) {
		return std::make_pair(node_count, branch_count);
	} else if (root->children.size() > 3) {
		throw std::runtime_error("The network is not bifurcating");
	}
	// special case: check if rnetwork.root has only one child... if so, reset the root to its child.
	while (root->children.size() == 1) {
		root = root->children[0];
		node_count--;
		branch_count--;
	}

	if (root->children.size() == 2) { // make it trifurcating
		unsigned int newRootChildIdx = 0;
		if (root->children[0]->children.size() == 0) {
			newRootChildIdx = 1;
		}
		RootedNetworkNode* new_root = root->children[newRootChildIdx];
		new_root->children.push_back(root->children[!newRootChildIdx]);
		root->children[!newRootChildIdx]->length += new_root->length;
		root->children[!newRootChildIdx]->parent = new_root;
		node_count--;
		branch_count--;
		root = new_root;
		root->parent = nullptr;
		rnetwork.root = root;
	}

	return std::make_pair(node_count, branch_count);
}

Network convertNetwork(RootedNetwork &rnetwork) {
	std::cout << exportDebugInfo(rnetwork) << "\n";
	std::pair<size_t, size_t> node_and_branch_count = makeToplevelTrifurcation(rnetwork);
	std::cout << exportDebugInfo(rnetwork) << "\n";
	size_t node_count = node_and_branch_count.first;
	size_t branch_count = node_and_branch_count.second;

	Network network;
	network = convertNetworkToplevelTrifurcation(rnetwork, node_count, branch_count);
	assert(!network.root->isTip());

	// ensure that no branch lengths are zero
	for (size_t i = 0; i < network.edges.size(); ++i) {
		assert(network.edges[i].length != 0);
	}
	assert(networkIsConnected(network));

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
