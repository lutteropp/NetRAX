/*
 * BiconnectedComponents.cpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#include "BiconnectedComponents.hpp"
#include "Edge.hpp"

#include <stack>
#include <cassert>
#include <limits>
#include <algorithm>

namespace netrax {

// Algorithm taken from https://www.cs.cmu.edu/~avrim/451f12/lectures/biconnected.pdf
void bicon(Node* v, Node* u, unsigned int& time, std::vector<unsigned int>& discovery_time,
		std::vector<unsigned int>& lowest, std::stack<Edge*>& s, std::vector<unsigned int>& edge_component_id,
		unsigned int& act_bicomp_id) {
	time++;
	discovery_time[v->clv_index] = time;
	lowest[v->clv_index] = time;
	for (Node* neigh : v->getNeighbors()) {
		if (discovery_time[neigh->clv_index] == 0) {
			// (v, neigh) is a forward edge
			s.push(v->getEdgeTo(neigh));
			bicon(neigh, v, time, discovery_time, lowest, s, edge_component_id, act_bicomp_id);
			lowest[v->clv_index] = std::min(lowest[v->clv_index], lowest[neigh->clv_index]);
			if (lowest[neigh->clv_index] >= discovery_time[v->clv_index]) {
				// v is either the root of the dfs tree or an articulation point.
				// Form a biconnected component consisting of all edges on the stack above and including (v, w).
				// Remove these edges from the stack.
				while (s.top() != v->getEdgeTo(neigh)) {
					Edge* actEdge = s.top();
					edge_component_id[actEdge->pmatrix_index] = act_bicomp_id;
					s.pop();
				}
				assert(s.top() = v->getEdgeTo(neigh));
				edge_component_id[s.top()->pmatrix_index] = act_bicomp_id;
				s.pop();
				act_bicomp_id++;
			}
		} else if (discovery_time[neigh->clv_index] < discovery_time[v->clv_index] && neigh != u) {
			// (v,neigh) is a back edge
			s.push(v->getEdgeTo(neigh));
			lowest[v->clv_index] = std::min(lowest[v->clv_index], discovery_time[neigh->clv_index]);
		}
	}
}

std::vector<unsigned int> partitionNetworkEdgesIntoBlobs(const Network& network) {
	std::vector<unsigned int> edge_component_id(network.num_edges(), std::numeric_limits<unsigned int>::max());
	unsigned int time = 0;
	unsigned int act_bicomp_id = 0;
	std::stack<Edge*> s;
	std::vector<unsigned int> discovery_time(network.num_nodes(), 0);
	std::vector<unsigned int> lowest(network.num_nodes(), std::numeric_limits<unsigned int>::max());
	for (Node* node : network.nodes) {
		if (discovery_time[node->clv_index] == 0) {
			bicon(node, nullptr, time, discovery_time, lowest, s, edge_component_id, act_bicomp_id);
		}
	}
	return edge_component_id;
}

std::vector<unsigned int> partitionNetworkNodesIntoBlobs(const Network& network) {
	std::vector<unsigned int> node_component_id(network.num_nodes(), std::numeric_limits<unsigned int>::max());
	std::vector<unsigned int> edge_component_id = partitionNetworkEdgesIntoBlobs(network);
	// ...


	throw std::runtime_error("This implementation is not finished yet");
	return node_component_id;
}

std::vector<unsigned int> partitionNetworkIntoBlobsSarah(const Network& network) {
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
	/* What we do is that we now search for connected components in the underlying undirected graph,
	 * but whenever we encounter an articulation point,
	 * we still take it into account for the edges, but don't add it to the stack.
	 */
	std::fill(visited.begin(), visited.end(), false);

	// ...

	throw std::runtime_error("This implementation is not finished yet");

	return node_component_id;
}
}
