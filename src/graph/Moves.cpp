/*
 * Moves.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#include "Moves.hpp"
#include <vector>
#include <queue>

namespace netrax {
    bool hasPath(const Network& network, const Node* from, const Node* to) {
        std::vector<bool> visited(network.num_nodes(), false);
        std::queue<const Node*> q;
        q.emplace(from);
        while (!q.empty()) {
            const Node* node = q.front();
            if (node == to) {
                return true;
            }
            q.pop();
            visited[node->clv_index] = true;
            for (const Node* neigh : node->getActiveChildren(node->getActiveParent())) {
                if (!visited[neigh->clv_index]) {
                    q.emplace(neigh);
                }
            }
        }
        return false;
    }
}