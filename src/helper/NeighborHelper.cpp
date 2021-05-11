#include "Helper.hpp"
#include <queue>

namespace netrax {

std::vector<Node*> getNeighbors(const Network &network, const Node *node) {
    assert(node);
    std::vector<Node*> neighbors;
    for (const auto &link : node->links) {
        Node *target = getTargetNode(network, &link);
        if (std::find(neighbors.begin(), neighbors.end(), target) == neighbors.end()) {
            neighbors.emplace_back(target);
        }
    }
    return neighbors;
}

std::vector<Node*> getActiveNeighbors(Network &network, const Node *node) {
    assert(node);
    std::vector<Node*> activeNeighbors;
    std::vector<Node*> neighbors = getNeighbors(network, node);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        if (neighbors[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the neighbor is active, this is, if we are currently the selected parent
            if (node != getReticulationChild(network, neighbors[i])
                    && getReticulationActiveParent(network, neighbors[i]) != node) {
                continue;
            }
        }
        if (node->getType() == NodeType::RETICULATION_NODE
                && neighbors[i] != getReticulationChild(network, node)) {
            if (neighbors[i] != getReticulationActiveParent(network, node)) {
                continue;
            }
        }
        activeNeighbors.push_back(neighbors[i]);
    }
    assert(activeNeighbors.size() <= 3);
    return activeNeighbors;
}

std::vector<Node*> getActiveAliveNeighbors(Network &network, const std::vector<bool> &dead_nodes,
        const Node *node) {
    assert(node);
    std::vector<Node*> activeNeighbors;
    std::vector<Node*> neighbors = getNeighbors(network, node);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        if (dead_nodes[neighbors[i]->clv_index]) {
            continue;
        }
        if (neighbors[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the neighbor is active, this is, if we are currently the selected parent
            if (node != getReticulationChild(network, neighbors[i])
                    && getReticulationActiveParent(network, neighbors[i]) != node) {
                continue;
            }
        }
        if (node->getType() == NodeType::RETICULATION_NODE
                && neighbors[i] != getReticulationChild(network, node)) {
            if (neighbors[i] != getReticulationActiveParent(network, node)) {
                continue;
            }
        }
        activeNeighbors.emplace_back(neighbors[i]);
    }
    assert(activeNeighbors.size() <= 3);
    return activeNeighbors;
}

bool hasNeighbor(Node *node1, Node *node2) {
    if (!node1 || !node2) {
        return false;
    }
    for (const auto &link : node1->links) {
        if (link.outer->node_clv_index == node2->clv_index) {
            return true;
        }
    }

    for (const auto &link : node2->links) {
        if (link.outer->node_clv_index == node1->clv_index) {
            throw std::runtime_error("The links are not symmetric");
        }
    }
    return false;
}

std::unordered_set<size_t> getNeighborPmatrixIndices(Network &network, Edge *edge) {
    assert(edge);
    std::unordered_set<size_t> res;
    Node *source = getSource(network, edge);
    Node *target = getTarget(network, edge);
    for (size_t i = 0; i < source->links.size(); ++i) {
        res.emplace(source->links[i].edge_pmatrix_index);
    }
    for (size_t i = 0; i < target->links.size(); ++i) {
        res.emplace(target->links[i].edge_pmatrix_index);
    }
    res.erase(edge->pmatrix_index);
    assert(
            res.size() == 2 || res.size() == 4 || (source == network.root && res.size() == 3)
                    || (source == network.root && target->isTip() && res.size() == 1));
    return res;
}

std::vector<Node*> getNeighborsWithinRadius(const Network& network, Node* node, size_t min_radius, size_t max_radius) {
    assert(min_radius <= max_radius);
    std::vector<Node*> res;
    std::vector<bool> seen(network.num_nodes(), false);
    std::queue<Node*> q;
    std::queue<Node*> next_q;
    q.emplace(node);
    size_t act_radius = 0;
    while (act_radius <= max_radius && !q.empty()) {
        while (!q.empty()) {
            Node* actNode = q.front();
            q.pop();
            seen[actNode->clv_index] = true;
            if (act_radius >= min_radius && act_radius <= max_radius) {
                res.emplace_back(actNode);
            }
            std::vector<Node*> neighs = getNeighbors(network, actNode);
            for (Node* neigh : neighs) {
                if (!seen[neigh->clv_index]) {
                    next_q.emplace(neigh);
                }
            }
        }
        act_radius++;
        std::swap(q, next_q);
    }
    return res;
}

}