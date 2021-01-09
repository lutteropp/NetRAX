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
    size_t num_tips() const;

    size_t num_inner() const;

    size_t num_branches() const;

    size_t num_reticulations() const;

    size_t num_nodes() const;

    Node* getNodeByLabel(const std::string &label);

    unsigned int nodeCount = 0;
    unsigned int branchCount = 0;
    unsigned int tipCount = 0;
    Node *root = nullptr;

    std::vector<Node*> nodes_by_index; // nodes by clv_index
    std::vector<Edge*> edges_by_index; // edges by pmatrix_index
    std::vector<Node*> reticulation_nodes;

    std::vector<Node> nodes;
    std::vector<Edge> edges;

    // reule of five
    Network() = default; // default constructor
    ~Network() = default; // destructor
    Network(Network&& other) noexcept { // move constructor
        this->nodeCount = other.nodeCount;
        this->branchCount = other.branchCount;
        this->tipCount = other.tipCount;
        std::swap(edges_by_index, other.edges_by_index);
        std::swap(nodes_by_index, other.nodes_by_index);
        std::swap(reticulation_nodes, other.reticulation_nodes);
        std::swap(nodes, other.nodes);
        std::swap(edges, other.edges);
        std::swap(root, other.root);
    }
    Network(const Network& other); // copy constructor
    Network& operator=(const Network& other) { // copy assignment
        return *this = Network(other);
    }
    Network& operator=(Network&& other) noexcept { // move assignment
        this->nodeCount = other.nodeCount;
        this->branchCount = other.branchCount;
        this->tipCount = other.tipCount;
        std::swap(edges_by_index, other.edges_by_index);
        std::swap(nodes_by_index, other.nodes_by_index);
        std::swap(reticulation_nodes, other.reticulation_nodes);
        std::swap(nodes, other.nodes);
        std::swap(edges, other.edges);
        std::swap(root, other.root);
        return *this;
    }
};

}