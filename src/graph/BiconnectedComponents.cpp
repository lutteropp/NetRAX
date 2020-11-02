/*
 * BiconnectedComponents.cpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#include "BiconnectedComponents.hpp"
#include "NetworkFunctions.hpp"
#include "NetworkTopology.hpp"
#include "Edge.hpp"
#include "Common.hpp"
#include "Node.hpp"

#include <stack>
#include <cassert>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <queue>

namespace netrax {

// Algorithm taken from https://www.cs.cmu.edu/~avrim/451f12/lectures/biconnected.pdf
void bicon(Network &network, const Node *v, const Node *u, unsigned int &time,
        std::vector<unsigned int> &discovery_time, std::vector<unsigned int> &lowest,
        std::stack<Edge*> &s, std::vector<unsigned int> &edge_component_id,
        unsigned int &act_bicomp_id, std::vector<unsigned int> &blob_sizes) {
    time++;
    discovery_time[v->clv_index] = time;
    lowest[v->clv_index] = time;
    for (Node *neigh : getNeighbors(network, v)) {
        if (discovery_time[neigh->clv_index] == 0) {
            // (v, neigh) is a forward edge
            s.push(getEdgeTo(network, v, neigh));
            bicon(network, neigh, v, time, discovery_time, lowest, s, edge_component_id,
                    act_bicomp_id, blob_sizes);
            lowest[v->clv_index] = std::min(lowest[v->clv_index], lowest[neigh->clv_index]);
            if (lowest[neigh->clv_index] >= discovery_time[v->clv_index]) {
                // v is either the root of the dfs tree or an articulation point.
                // Form a biconnected component consisting of all edges on the stack above and including (v, w).
                // Remove these edges from the stack.
                unsigned int act_blob_size = 0;
                while (s.top() != getEdgeTo(network, v, neigh)) {
                    Edge *actEdge = s.top();
                    edge_component_id[actEdge->pmatrix_index] = act_bicomp_id;
                    s.pop();
                    act_blob_size++;
                }
                assert(s.top() = getEdgeTo(network, v, neigh));
                edge_component_id[s.top()->pmatrix_index] = act_bicomp_id;
                s.pop();
                act_blob_size++;
                blob_sizes.emplace_back(act_blob_size);
                act_bicomp_id++;
                assert(blob_sizes.size() == act_bicomp_id);
            }
        } else if (discovery_time[neigh->clv_index] < discovery_time[v->clv_index] && neigh != u) {
            // (v,neigh) is a back edge
            s.push(getEdgeTo(network, v, neigh));
            lowest[v->clv_index] = std::min(lowest[v->clv_index], discovery_time[neigh->clv_index]);
        }
    }
}

unsigned int get_node_blob_id(Network &network, Node *node, const BlobInformation &blobInfo,
        const std::vector<Node*> parent) {
    unsigned int res = std::numeric_limits<unsigned int>::infinity();
    std::unordered_set<unsigned int> seen;
    for (Node *neigh : getNeighbors(network, node)) {
        unsigned int actBlobID =
                blobInfo.edge_blob_id[getEdgeTo(network, node, neigh)->pmatrix_index];
        if (seen.find(actBlobID) != seen.end()) {
            res = actBlobID;
        }
        seen.emplace(actBlobID);
    };

    if (res == std::numeric_limits<unsigned int>::infinity()) {
        if (parent[node->clv_index] != nullptr) { // catch the network root node
            unsigned int edgeToParentBlobID = blobInfo.edge_blob_id[getEdgeTo(network, node,
                    parent[node->clv_index])->pmatrix_index];
            res = edgeToParentBlobID;
        }
    }
    return res;
}

void gather_reticulations_per_megablob(Network &network, BlobInformation &blob_info) {
    // Given the megablob roots at blobInfo.megablob_roots
    // Fill blobInfo.reticulation_nodes_per_megablob
    // ... BFS traversal? Always taking the current megablob index with us...
    std::queue<std::pair<Node*, unsigned int> > q;
    std::vector<bool> visited(network.nodes.size(), false);
    q.emplace(std::make_pair(network.root, blob_info.megablob_roots.size() - 1));
    while (!q.empty()) {
        auto entry = q.front();
        q.pop();
        if (visited[entry.first->clv_index]) {
            continue;
        }
        visited[entry.first->clv_index] = true;
        unsigned int act_megablob_index = entry.second;
        if (entry.first->type == NodeType::RETICULATION_NODE) {
            blob_info.reticulation_nodes_per_megablob[act_megablob_index].emplace_back(entry.first);
        } else {
            auto it = std::find(blob_info.megablob_roots.begin(), blob_info.megablob_roots.end(),
                    entry.first);
            if (it != blob_info.megablob_roots.end()) {
                act_megablob_index = std::distance(blob_info.megablob_roots.begin(), it);
            }
        }
        for (Node *child : getChildren(network, entry.first)) {
            if (!visited[child->clv_index]) {
                q.emplace(std::make_pair(child, act_megablob_index));
            }
        }
    }

    // some sanity assertion
    size_t retsum = 0;
    for (size_t i = 0; i < blob_info.reticulation_nodes_per_megablob.size(); ++i) {
        retsum += blob_info.reticulation_nodes_per_megablob[i].size();
    }
    assert(retsum == network.num_reticulations());

    for (size_t i = 0; i < blob_info.reticulation_nodes_per_megablob.size(); ++i) {
        std::sort(blob_info.reticulation_nodes_per_megablob[i].begin(),
                blob_info.reticulation_nodes_per_megablob[i].end(),
                [](const Node *a, const Node *b) {
                    return a->clv_index < b->clv_index;
                });
    }
}

BlobInformation partitionNetworkIntoBlobs(Network &network, const std::vector<Node*> &travbuffer) {
    BlobInformation blob_info;
    blob_info.edge_blob_id.resize(network.edges.size());
    unsigned int time = 0;
    unsigned int act_bicomp_id = 0;
    std::stack<Edge*> s;
    std::vector<unsigned int> discovery_time(network.nodes.size(), 0);
    std::vector<unsigned int> lowest(network.nodes.size(),
            std::numeric_limits<unsigned int>::max());
    std::vector<unsigned int> blob_size;
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        Node *node = &network.nodes[i];
        if (discovery_time[node->clv_index] == 0) {
            bicon(network, node, nullptr, time, discovery_time, lowest, s, blob_info.edge_blob_id,
                    act_bicomp_id, blob_size);
        }
    }

    std::vector<Node*> parent = grab_current_node_parents(network);
    // fill the megablob roots now.
    // also, put the reticulation nodes into their megablobs

    // A node is a megablob root, if and only if:
    // 1.) It is the root node, or
    // 2.) The parent has another blob id

    blob_info.reticulation_nodes_per_megablob.emplace_back(std::vector<Node*>());
    std::vector<unsigned int> node_blob_id(network.nodes.size());
    for (size_t i = 0; i < travbuffer.size(); ++i) {
        node_blob_id[travbuffer[i]->clv_index] = get_node_blob_id(network, travbuffer[i], blob_info,
                parent);
    }

    blob_info.node_blob_id = node_blob_id;

    for (size_t i = 0; i < travbuffer.size() - 1; ++i) {
        Node *node = travbuffer[i];
        if (node->isTip())
            continue; // no need to make a megablob root out of a tip node

        unsigned int blobId = node_blob_id[travbuffer[i]->clv_index];
        unsigned int parentBlobId = node_blob_id[parent[travbuffer[i]->clv_index]->clv_index];

        if (blobId != parentBlobId) {
            blob_info.megablob_roots.emplace_back(travbuffer[i]);
            blob_info.reticulation_nodes_per_megablob.emplace_back(std::vector<Node*>());
        }
    }
    blob_info.megablob_roots.emplace_back(network.root);
    assert(blob_info.megablob_roots.size() == blob_info.reticulation_nodes_per_megablob.size());

    gather_reticulations_per_megablob(network, blob_info);


    // just for debug
    blob_info.megablob_roots = {network.root};
    blob_info.reticulation_nodes_per_megablob = {network.reticulation_nodes};

    //std::cout << "Network for debug:\n";
    //std::cout << exportDebugInfoBlobs(network, blob_info) << "\n";

    return blob_info;
}
}
