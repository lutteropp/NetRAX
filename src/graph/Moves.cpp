/*
 * Moves.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#include "Moves.hpp"
#include "NetworkTopology.hpp"
#include <vector>
#include <queue>

namespace netrax {
    bool hasPath(const Network& network, const Node* from, const Node* to, bool nonelementary = false) {
        std::vector<bool> visited(network.num_nodes(), false);
        std::queue<std::pair<const Node*, const Node*> > q;
        q.emplace(to, nullptr);
        while (!q.empty()) {
            const Node* node = q.front().first;
            const Node* child = q.front().second;
            if (node == from) {
            	if (!nonelementary || child != to) {
            		return true;
            	}
            }
            q.pop();
            visited[node->clv_index] = true;
            for (const Node* neigh : getAllParents(node)) {
                if (!visited[neigh->clv_index]) {
                    q.emplace(std::make_pair(neigh, node));
                }
            }
        }
        return false;
    }

	/*
	 * we need to choose s and t in a way that there are elementary connections {u,s} and {v,t},
	 * but there are no elementary connections {u,t} and {v,s}
	*/
    std::vector<std::pair<Node*, Node*> > getSTChoices(const Edge& edge) {
    	std::vector<std::pair<Node*, Node*> > res;
    	Node* u = getSource(edge);
    	Node* v = getTarget(edge);

    	auto uNeighbors = getNeighbors(u);
    	auto vNeighbors = getNeighbors(v);

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

    bool isOutgoing(const Node* from, const Node* to) {
    	auto children = getChildren(from, getActiveParent(from));
    	return (std::find(children.begin(), children.end(), to) != children.end());
    }

    std::vector<RNNIMove> possibleRNNIMoves(const Network& network, const Edge& edge) {
    	std::vector<RNNIMove> res;
    	Node* u = getSource(edge);
    	Node* v = getTarget(edge);
    	auto stChoices = getSTChoices(edge);
    	for (const auto& st : stChoices) {
    		Node* s = st.first;
    		Node* t = st.second;

    		// check for possible variant and add move from the paper if the move would not create a cycle
    		if (isOutgoing(u, s) && isOutgoing(v, t)) {
    			if (!hasPath(network, s, v)) {
    				// add move 1
    				res.emplace_back(RNNIMove{u, v, s, t, RNNIMoveType::ONE});
    				if (v->type == NodeType::RETICULATION_NODE) {
						// add move 1*
    					res.emplace_back(RNNIMove{u, v, s, t, RNNIMoveType::ONE_STAR});
					}
    			}
    		} else if (isOutgoing(s, u) && isOutgoing(t, v)) {
    			if (!hasPath(network, u, t)) {
    				// add move 2
    				res.emplace_back(RNNIMove{u, v, s, t, RNNIMoveType::TWO});
    				if (u->type != NodeType::RETICULATION_NODE) {
						// add move 2*
    					res.emplace_back(RNNIMove{u, v, s, t, RNNIMoveType::TWO_STAR});
					}
    			}
    		} else if (isOutgoing(s, u) && isOutgoing(v, t)) {
    			if (u->type == NodeType::RETICULATION_NODE && v->type != NodeType::RETICULATION_NODE) {
    				// add move 3
    				res.emplace_back(RNNIMove{u, v, s, t, RNNIMoveType::THREE});
    			}
    			if (!hasPath(network, u, v, true)) {
    				// add move 3*
    				res.emplace_back(RNNIMove{u, v, s, t, RNNIMoveType::THREE_STAR});
    			}
    		} else if (isOutgoing(u, s) && isOutgoing(t, v)) {
    			if (!hasPath(network, s, t)) {
    				// add move 4
    				res.emplace_back(RNNIMove{u, v, s, t, RNNIMoveType::FOUR});
    			}
    		}
    	}
    	return res;
    }

    void performMove(Network& network, RNNIMove& move) {

    	throw std::runtime_error("Not implemented yet");
    }

    void undoMove(Network&network, RNNIMove& move) {
    	throw std::runtime_error("Not implemented yet");
    }
}
