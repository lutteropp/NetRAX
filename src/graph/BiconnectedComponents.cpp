/*
 * BiconnectedComponents.cpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#include "BiconnectedComponents.hpp"
#include "NetworkFunctions.hpp"
#include "Edge.hpp"
#include "Common.hpp"
#include "Node.hpp"

#include <stack>
#include <cassert>
#include <limits>
#include <algorithm>

namespace netrax {

// Algorithm taken from https://www.cs.cmu.edu/~avrim/451f12/lectures/biconnected.pdf
void bicon(const Node* v, const Node* u, unsigned int& time, std::vector<unsigned int>& discovery_time, std::vector<unsigned int>& lowest,
		std::stack<Edge*>& s, std::vector<unsigned int>& edge_component_id, unsigned int& act_bicomp_id,
		std::vector<unsigned int>& blob_sizes) {
	time++;
	discovery_time[v->clv_index] = time;
	lowest[v->clv_index] = time;
	for (Node* neigh : v->getNeighbors()) {
		if (discovery_time[neigh->clv_index] == 0) {
			// (v, neigh) is a forward edge
			s.push(v->getEdgeTo(neigh));
			bicon(neigh, v, time, discovery_time, lowest, s, edge_component_id, act_bicomp_id, blob_sizes);
			lowest[v->clv_index] = std::min(lowest[v->clv_index], lowest[neigh->clv_index]);
			if (lowest[neigh->clv_index] >= discovery_time[v->clv_index]) {
				// v is either the root of the dfs tree or an articulation point.
				// Form a biconnected component consisting of all edges on the stack above and including (v, w).
				// Remove these edges from the stack.
				unsigned int act_blob_size = 0;
				while (s.top() != v->getEdgeTo(neigh)) {
					Edge* actEdge = s.top();
					edge_component_id[actEdge->pmatrix_index] = act_bicomp_id;
					s.pop();
					act_blob_size++;
				}
				assert(s.top() = v->getEdgeTo(neigh));
				edge_component_id[s.top()->pmatrix_index] = act_bicomp_id;
				s.pop();
				act_blob_size++;
				blob_sizes.emplace_back(act_blob_size);
				act_bicomp_id++;
				assert(blob_sizes.size() == act_bicomp_id);
			}
		} else if (discovery_time[neigh->clv_index] < discovery_time[v->clv_index] && neigh != u) {
			// (v,neigh) is a back edge
			s.push(v->getEdgeTo(neigh));
			lowest[v->clv_index] = std::min(lowest[v->clv_index], discovery_time[neigh->clv_index]);
		}
	}
}

BlobInformation partitionNetworkIntoBlobs(const Network& network) {
	BlobInformation blob_info { std::vector<unsigned int>(network.num_edges(), std::numeric_limits<unsigned int>::max()),
							    std::vector<unsigned int>() };
	unsigned int time = 0;
	unsigned int act_bicomp_id = 0;
	std::stack<Edge*> s;
	std::vector<unsigned int> discovery_time(network.num_nodes(), 0);
	std::vector<unsigned int> lowest(network.num_nodes(), std::numeric_limits<unsigned int>::max());
	for (const Node& node : network.nodes) {
		if (discovery_time[node.clv_index] == 0) {
			bicon(&node, nullptr, time, discovery_time, lowest, s, blob_info.edge_blob_id, act_bicomp_id, blob_info.blob_size);
		}
	}

	blob_info.reticulation_nodes_per_blob.resize(blob_info.blob_size.size());
	for (size_t i = 0; i < network.num_reticulations(); ++i) {
		const Node* retNode = network.reticulation_nodes[i];
		unsigned int firstParentEdgeIndex = retNode->getReticulationData()->getLinkToFirstParent()->edge->pmatrix_index;
		unsigned int ret_blob_id = blob_info.edge_blob_id[firstParentEdgeIndex];
		blob_info.reticulation_nodes_per_blob[ret_blob_id].emplace_back(retNode);
	}

	return blob_info;
}
}
