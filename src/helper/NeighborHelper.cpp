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

bool hasNeighbor(const Node *node1, const Node *node2) {
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

std::unordered_set<size_t> getNeighborPmatrixIndices(Network &network, const Edge *edge) {
    assert(edge);
    std::unordered_set<size_t> res;
    const Node *source = getSource(network, edge);
    const Node *target = getTarget(network, edge);
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

std::vector<Node*> getNeighborsWithinRadius(const Network& network, const Node* node, int min_radius, int max_radius) {
    assert(min_radius <= max_radius);
    if (min_radius == 0 && max_radius == std::numeric_limits<int>::max()) {
        std::vector<Node*> quick_res;
        for (size_t i = 0; i < network.num_nodes(); ++i) {
            Node* act_node = network.nodes_by_index[i];
            if (act_node != node) {
                quick_res.emplace_back(act_node);
            }
        }
        return quick_res;
    }

    std::vector<Node*> res;
    std::vector<bool> seen(network.num_nodes(), false);
    std::queue<Node*> q;
    std::queue<Node*> next_q;
    q.emplace(network.nodes_by_index[node->clv_index]);
    int act_radius = 0;
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

bool topology_equal(Network& n1, Network& n2) {
    if (n1.num_branches() != n2.num_branches()) {
        std::cout << "topology not equal: different num branches \n";
        return false;
    }
    for (size_t i = 0; i < n1.num_branches(); ++i) {
        if (getSource(n1, n1.edges_by_index[i])->clv_index != getSource(n2, n2.edges_by_index[i])->clv_index) {
            std::cout << "topology not equal\n";
            return false;
        }
        if (getTarget(n1, n1.edges_by_index[i])->clv_index != getTarget(n2, n2.edges_by_index[i])->clv_index) {
            std::cout << "topology not equal\n";
            return false;
        }
    }
    if (n1.root->clv_index != n2.root->clv_index) {
        std::cout << "edges are fine, but root is wrong\n";
        return false;
    }
    return true;
}

}