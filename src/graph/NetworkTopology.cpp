/*
 * NetworkTopology.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: sarah
 */

#include "NetworkTopology.hpp"
#include "Node.hpp"

namespace netrax {

Node* getTargetNode(const Link *link) {
    assert(link);
    if (link->edge->link1 == link) {
        return link->edge->link2->node;
    } else {
        assert(link->edge->link2 == link);
        return link->edge->link1->node;
    }
}

Link* getLinkToClvIndex(Node *node, size_t target_index) {
    assert(node);
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(&(node->links[i]))->clv_index == target_index) {
            return &(node->links[i]);
        }
    }
    return nullptr;
}

Node* getReticulationChild(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(node->getReticulationData()->getLinkToChild());
}

Node* getReticulationFirstParent(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(node->getReticulationData()->link_to_first_parent);
}

Node* getReticulationSecondParent(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(node->getReticulationData()->link_to_second_parent);
}

Node* getReticulationActiveParent(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(node->getReticulationData()->getLinkToActiveParent());
}

std::vector<Node*> getChildren(const Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> children;
    if (node->type == NodeType::RETICULATION_NODE) {
        children.push_back(getReticulationChild(node));
    } else { // normal node
        std::vector<Node*> neighbors = getNeighbors(node);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (neighbors[i] != myParent) {
                children.push_back(neighbors[i]);
            }
        }
    }
    return children;
}

std::vector<Node*> getActiveChildren(const Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> activeChildren;
    std::vector<Node*> children = getChildren(node, myParent);
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the child is active, this is, if we are currently the selected parent
            if (getActiveParent(children[i]) != node) {
                continue;
            }
        }
        activeChildren.push_back(children[i]);
    }
    assert(activeChildren.size() <= 2 || (myParent == nullptr && activeChildren.size() == 3));
    return activeChildren;
}

std::vector<Node*> getNeighbors(const Node *node) {
    assert(node);
    std::vector<Node*> neighbors;
    for (const auto &link : node->links) {
        Node *target = getTargetNode(&link);
        neighbors.push_back(target);
    }
    return neighbors;
}

std::vector<Node*> getActiveNeighbors(const Node *node) {
    assert(node);
    std::vector<Node*> activeNeighbors;
    std::vector<Node*> neighbors = getNeighbors(node);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        if (neighbors[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the neighbor is active, this is, if we are currently the selected parent
            if (getReticulationActiveParent(neighbors[i]) != node) {
                continue;
            }
        }
        activeNeighbors.push_back(neighbors[i]);
    }
    assert(activeNeighbors.size() <= 3);
    return activeNeighbors;
}

Node* getActiveParent(const Node *node) {
    assert(node);
    if (node->type == NodeType::RETICULATION_NODE) {
        return getReticulationActiveParent(node);
    }
    for (const auto &link : node->links) {
        if (link.direction == Direction::INCOMING) {
            return getTargetNode(&link);
        }
    }
    return nullptr;
}

std::vector<Node*> getAllParents(const Node *node) {
    assert(node);
    std::vector<Node*> res;
    if (node->type == NodeType::RETICULATION_NODE) {
        res.emplace_back(getReticulationFirstParent(node));
        res.emplace_back(getReticulationSecondParent(node));
    } else {
        Node *parent = getActiveParent(node);
        if (parent) {
            res.emplace_back(parent);
        }
    }
    return res;
}

Edge* getEdgeTo(const Node *node, const Node *target) {
    assert(node);
    assert(target);
    for (const auto &link : node->links) {
        if (link.outer->node == target) {
            return link.edge;
        }
    }
    throw std::runtime_error("The given target node is not a neighbor of this node");
}

Node* getSource(const Edge& edge) {
    if (edge.link1->direction == Direction::OUTGOING) {
        return edge.link1->node;
    } else {
        return edge.link2->node;
    }
}

Node* getTarget(const Edge& edge) {
    if (edge.link1->direction == Direction::INCOMING) {
        return edge.link1->node;
    } else {
        return edge.link2->node;
    }
}

}
