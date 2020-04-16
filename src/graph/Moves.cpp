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
        q.emplace(to);
        while (!q.empty()) {
            const Node* node = q.front();
            if (node == from) {
                return true;
            }
            q.pop();
            visited[node->clv_index] = true;
            for (const Node* neigh : node->getAllParents()) {
                if (!visited[neigh->clv_index]) {
                    q.emplace(neigh);
                }
            }
        }
        return false;
    }

	/*
	 * we need to choose s and t in a way that there are connections {u,s} and {v,t},
	 * but there are no connections {u,t} and {v,s}
	*/
    std::vector<std::pair<Node*, Node*> > getSTChoices(const Edge& edge) {
    	std::vector<std::pair<Node*, Node*> > res;
    	Node* u = edge.getSource();
    	Node* v = edge.getTarget();

    	auto uNeighbors = u->getNeighbors();
    	auto vNeighbors = v->getNeighbors();

    	for (const auto& s : uNeighbors) {
    		if (s == v) continue;
    		for (const auto& t : vNeighbors) {
    			if (t == u) continue;

    			if (std::find(uNeighbors.begin(), uNeighbors.end(), t) == uNeighbors.end()
    					&& std::find(vNeighbors.begin(), vNeighbors.end(), s) == vNeighbors.end()) {
    				res.emplace_back(std::make_pair(s, t));
    			}
    		}
    	}
    	return res;
    }

    std::vector<RNNIMove> possibleRNNIMoves(const Network& network, const Edge& edge) {
    	std::vector<RNNIMove> res;



    	throw std::runtime_error("Not implemented yet");
    	return res;
    }

    void performMove(Network& network, RNNIMove& move, Edge& edge) {
    	Node* u = edge.getSource();
    	Node* v = edge.getTarget();

    	throw std::runtime_error("Not implemented yet");
    }

    void undoMove(Network&network, RNNIMove& move, Edge& edge) {
    	throw std::runtime_error("Not implemented yet");
    }
}
