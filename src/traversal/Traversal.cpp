/*
 * Traversal.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "Traversal.hpp"

#include "../helper/NetworkTopology.hpp"
#include "../helper/NetworkFunctions.hpp"

namespace netrax {

void postorder(Network &network, Node *parent, Node *actNode, std::vector<Node*> &buffer) {
    if (actNode->getType() == NodeType::BASIC_NODE) { // visit the two children
        std::vector<Node*> neighbors = getNeighbors(network, actNode);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (neighbors[i] != parent) {
                // check if the current neighbor is a reticulation node, if so, only go this path if we are the active parent
                if (neighbors[i]->getType() != NodeType::RETICULATION_NODE
                        || getReticulationActiveParent(network, neighbors[i]) == actNode) {
                    postorder(network, actNode, neighbors[i], buffer);
                }
            }
        }
    } else { // only visit the child node
        postorder(network, actNode, getReticulationChild(network, actNode), buffer);
    }

    // 2) visit the node itself
    buffer.push_back(actNode);
}

std::vector<Node*> postorderTraversal(Network &network, size_t tree_index) {
    std::vector<Node*> buffer;
    setReticulationParents(network, tree_index);
    postorder(network, nullptr, network.root, buffer);
    return buffer;
}

}
