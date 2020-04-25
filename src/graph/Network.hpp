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
    Network() :
            root(nullptr) {
    }

    size_t num_tips() const {
        return tip_nodes.size();
    }

    size_t num_inner() const {
        return nodes.size() - tip_nodes.size();
    }

    size_t num_branches() const {
        return edges.size();
    }

    size_t num_reticulations() const {
        return reticulation_nodes.size();
    }

    size_t num_nodes() const {
        return nodes.size();
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

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Link*> links;
    std::vector<Node*> reticulation_nodes;
    std::vector<Node*> tip_nodes;
    std::vector<Node*> inner_nodes;
    Node *root;
};

}
