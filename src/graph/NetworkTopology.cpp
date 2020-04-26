/*
 * NetworkTopology.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: sarah
 */

#include "NetworkTopology.hpp"
#include "Node.hpp"
#include "Link.hpp"
#include "Edge.hpp"

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

bool isOutgoing(Node *from, Node *to) {
    assert(getLinkToClvIndex(from, to->clv_index));
    auto children = getChildren(from, getActiveParent(from));
    return (std::find(children.begin(), children.end(), to) != children.end());
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

Link* getLinkToNode(Node *node, Node *target) {
    assert(node);
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(&(node->links[i])) == target) {
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

double getReticulationFirstParentProb(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    assert(
            node->getReticulationData()->link_to_first_parent->edge->prob
                    + node->getReticulationData()->link_to_second_parent->edge->prob == 1.0);
    return node->getReticulationData()->link_to_first_parent->edge->prob;
}
double getReticulationSecondParentProb(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    assert(
            node->getReticulationData()->link_to_first_parent->edge->prob
                    + node->getReticulationData()->link_to_second_parent->edge->prob == 1.0);
    return node->getReticulationData()->link_to_second_parent->edge->prob;
}

double getReticulationActiveProb(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    assert(
            node->getReticulationData()->link_to_first_parent->edge->prob
                    + node->getReticulationData()->link_to_second_parent->edge->prob == 1.0);
    return node->getReticulationData()->getLinkToActiveParent()->edge->prob;
}

size_t getReticulationFirstParentPmatrixIndex(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToFirstParent()->edge->pmatrix_index;
}

size_t getReticulationSecondParentPmatrixIndex(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToSecondParent()->edge->pmatrix_index;
}

size_t getReticulationActiveParentPmatrixIndex(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToActiveParent()->edge->pmatrix_index;
}

std::vector<Node*> getChildren(Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> children;
    if (node->type == NodeType::RETICULATION_NODE) {
        children.push_back(getReticulationChild(node));
    } else { // normal node
        std::vector<Node*> neighbors = getNeighbors(node);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (neighbors[i] != myParent) {
                // TODO: The following IF destroys the unit test, but it has to be in here in order to correctly return the children
                if (getLinkToClvIndex(node, neighbors[i]->clv_index)->direction == Direction::OUTGOING) {
                    children.push_back(neighbors[i]);
                }
            }
        }
    }

    return children;
}

std::vector<Node*> getActiveChildren(Node *node, const Node *myParent) {
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

Node* getOtherChild(Node* parent, Node* aChild) {
    assert(parent && parent->type != NodeType::RETICULATION_NODE);
    assert(aChild);
    std::vector<Node*> children = getChildren(parent, getActiveParent(parent));
    assert(std::find(children.begin(), children.end(), aChild) != children.end());
    for (Node* child : children) {
        if (child != aChild) {
            return child;
        }
    }
    return nullptr;
}

bool hasChild(Node* parent, Node* candidate) {
    assert(parent);
    assert(candidate);
    std::vector<Node*> children = getChildren(parent, getActiveParent(parent));
    return (std::find(children.begin(), children.end(), candidate) != children.end());
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

std::vector<Edge*> getAdjacentEdges(const Node *node) {
    std::vector<Edge*> res;
    std::vector<Node*> neighs = getNeighbors(node);
    for (size_t i = 0; i < neighs.size(); ++i) {
        res.emplace_back(getEdgeTo(node, neighs[i]));
    }
    assert(res.size() <= 3);
    return res;
}

Node* getSource(const Edge *edge) {
    if (edge->link1->direction == Direction::OUTGOING) {
        return edge->link1->node;
    } else {
        return edge->link2->node;
    }
}

Node* getTarget(const Edge *edge) {
    if (edge->link1->direction == Direction::INCOMING) {
        return edge->link1->node;
    } else {
        return edge->link2->node;
    }
}

bool hasNeighbor(Node *node1, Node *node2) {
    if (!node1 || !node2) {
        return false;
    }
    for (const auto &link : node1->links) {
        if (link.outer->node == node2) {
            return true;
        }
    }
    return false;
}

}
