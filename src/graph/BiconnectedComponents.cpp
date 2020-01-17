/*
 * BiconnectedComponents.cpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#include "BiconnectedComponents.hpp"

#include <stack>
#include <cassert>
#include <limits>

namespace netrax {
std::vector<unsigned int> partitionNetworkIntoBlobs(const Network& network) {
	/* As a phylogenetic network is always connected by definition,
	no need for an outer loops that calls dfs on each still unvisited node.*/
	std::vector<unsigned int> node_component_id(network.num_nodes(), std::numeric_limits<unsigned int>::max());
	std::vector<unsigned int> edge_component_id(network.num_edges(), std::numeric_limits<unsigned int>::max());
	std::vector<bool> is_articulation_point(network.num_nodes(), false);

	// do a stack-based iterative dfs
	std::stack<Node*> s;
	std::vector<bool> visited(network.num_nodes(), false);
	std::vector<unsigned int> discovery_time(network.num_nodes(), std::numeric_limits<unsigned int>::max());
	std::vector<unsigned int> lowest(network.num_nodes(), std::numeric_limits<unsigned int>::max());
	std::vector<Node*> parent(network.num_nodes(), nullptr);
	assert(network.root);
	s.push(network.root);
	unsigned int time = 0;

	while (!s.empty()) {
		time++;
		Node* actNode = s.top();
		s.pop();
		discovery_time[actNode->clv_index] = time;
		lowest[actNode->clv_index] = std::min(lowest[actNode->clv_index], time);
		visited[actNode->clv_index] = true;
		for (Node* neigh : actNode->getNeighbors()) {
			if (!visited[neigh->clv_index]) {
				s.push(neigh);
				parent[neigh->clv_index] = actNode;
			}
			lowest[actNode->clv_index] = std::min(lowest[actNode->clv_index], discovery_time[neigh->clv_index]);
			lowest[neigh->clv_index] = std::min(lowest[neigh->clv_index], lowest[actNode->clv_index]);
		}
	}

	// find articulation points
	for (size_t i = 0; i < network.num_nodes(); ++i) {
		Node* node = network.nodes[i];
		Node* parent_node = parent[node->clv_index];
		if (parent[node->clv_index] != nullptr) {
			if (lowest[node->clv_index] >= discovery_time[parent_node->clv_index]) {
				is_articulation_point[parent_node->clv_index] = true;
			}
		}
	}
	// check number of dfs-forward-edges for the root node
	unsigned int root_forward_edge_cnt = 0;
	for (size_t i = 0; i < network.num_nodes(); ++i) {
		Node* actNode = network.nodes[i];
		if (parent[actNode->clv_index] == network.root->clv_index) {
			root_forward_edge_cnt++;
		}
	}
	if (root_forward_edge_cnt >= 2) {
		is_articulation_point[network.root->clv_index] = true;
	}

	// TODO: Given articulation points, partition the graph into blobs.

	// ...

	throw std::runtime_error("This implementation is not finished yet");

	return node_component_id;
}
}
