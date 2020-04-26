/*
 * Network.hpp
 *
 *  Created on: Sep 2, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <cassert>
#include <string>
#include <vector>

#include "Edge.hpp"
#include "Link.hpp"
#include "Node.hpp"

namespace netrax {

class Network {
public:
    size_t num_tips() const {
        return tipCount;
    }

    size_t num_inner() const {
        return nodeCount - tipCount;
    }

    size_t num_branches() const {
        return branchCount;
    }

    size_t num_reticulations() const {
        return reticulation_nodes.size();
    }

    size_t num_nodes() const {
        return nodeCount;
    }

    Node* getNodeByLabel(const std::string &label) {
        Node *result = nullptr;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].getLabel() == label) {
                result = &nodes[i];
                break;
            }
        }
        assert(result);
        return result;
    }

    const Node* getNodeByClvIndex(size_t idx) const {
        const Node *result = nullptr;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].clv_index == idx) {
                result = &nodes[i];
                break;
            }
        }
        assert(result);
        return result;
    }

    Node* getNodeByClvIndex(size_t idx) {
        Node *result = nullptr;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].clv_index == idx) {
                result = &nodes[i];
                break;
            }
        }
        assert(result);
        return result;
    }

    unsigned int nodeCount = 0;
    unsigned int branchCount = 0;
    unsigned int tipCount = 0;
    Node *root = nullptr;
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Link*> links;
    std::vector<Node*> reticulation_nodes;
};

}
