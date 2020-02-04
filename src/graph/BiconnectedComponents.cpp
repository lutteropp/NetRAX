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
#include <unordered_set>

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

unsigned int get_node_blob_id(const Node* node, const BlobInformation& blobInfo, const std::vector<const Node*> parent) {
	unsigned int res = std::numeric_limits<unsigned int>::infinity();
	std::unordered_set<unsigned int> seen;
	for (Node* neigh : node->getNeighbors()) {
		unsigned int actBlobID = blobInfo.edge_blob_id[node->getEdgeTo(neigh)->pmatrix_index];
		if (seen.find(actBlobID) != seen.end()) {
			res = actBlobID;
		}
		seen.emplace(actBlobID);
	};

	if (res == std::numeric_limits<unsigned int>::infinity()) {
		if (parent[node->clv_index] != nullptr) { // catch the network root node
			unsigned int edgeToParentBlobID = blobInfo.edge_blob_id[node->getEdgeTo(parent[node->clv_index])->pmatrix_index];
			res = edgeToParentBlobID;
		}
	}
	return res;
}

BlobInformation partitionNetworkIntoBlobs(const Network& network) {
	BlobInformation blob_info { std::vector<unsigned int>(network.num_edges(), std::numeric_limits<unsigned int>::max()), std::vector<
			unsigned int>(), std::vector<std::vector<const Node*> >(), std::vector<const Node*>() };
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

	std::vector<const Node*> parent = grab_current_node_parents(network);
	// fill the megablob roots now.
	std::vector<const Node*> travbuffer = netrax::reversed_topological_sort(network);
	unsigned int lastBlobId = std::numeric_limits<unsigned int>::infinity();
	for (size_t i = 0; i < travbuffer.size(); ++i) {
		const Node* node = travbuffer[i];
		unsigned int blobId = get_node_blob_id(node, blob_info, parent);
		if (blobId != lastBlobId && i > 0) {
			if (blob_info.blob_size[lastBlobId] == 1 && blob_info.blob_size[blobId] > 1) {
				blob_info.megablob_roots.emplace_back(travbuffer[i - 1]);
			}
		}
		lastBlobId = blobId;
	}
	blob_info.megablob_roots.emplace_back(network.root);

	return blob_info;
}
}
