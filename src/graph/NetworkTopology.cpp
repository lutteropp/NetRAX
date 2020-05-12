/*
 * NetworkTopology.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: sarah
 */

#include <iostream>

#include "NetworkTopology.hpp"
#include "Node.hpp"
#include "Link.hpp"
#include "Edge.hpp"
#include "Network.hpp"

namespace netrax {

Node* getTargetNode(Network &network, const Link *link) {
    assert(link);
    if (network.edges_by_index[link->edge_pmatrix_index]->link1 == link) {
        return network.nodes_by_index[network.edges_by_index[link->edge_pmatrix_index]->link2->node_clv_index];
    } else {
        assert(network.edges_by_index[link->edge_pmatrix_index]->link2 == link);
        return network.nodes_by_index[network.edges_by_index[link->edge_pmatrix_index]->link1->node_clv_index];
    }
}

bool isOutgoing(Network &network, Node *from, Node *to) {
    assert(getLinkToClvIndex(network, from, to->clv_index));
    auto children = getChildren(network, from);
    return (std::find(children.begin(), children.end(), to) != children.end());
}

Link* getLinkToClvIndex(Network &network, Node *node, size_t target_index) {
    assert(node);
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(network, &(node->links[i]))->clv_index == target_index) {
            return &(node->links[i]);
        }
    }
    return nullptr;
}

Link* getLinkToNode(Network &network, Node *node, Node *target) {
    assert(node);
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(network, &(node->links[i])) == target) {
            return &(node->links[i]);
        }
    }
    return nullptr;
}

Link* getLinkToNode(Network &network, size_t from_clv_index, size_t to_clv_index) {
    return getLinkToNode(network, network.nodes_by_index[from_clv_index], network.nodes_by_index[to_clv_index]);
}

Node* getReticulationChild(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(network, node->getReticulationData()->getLinkToChild());
}

Node* getReticulationFirstParent(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(network, node->getReticulationData()->link_to_first_parent);
}

Node* getReticulationSecondParent(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(network, node->getReticulationData()->link_to_second_parent);
}

Node* getReticulationActiveParent(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(network, node->getReticulationData()->getLinkToActiveParent());
}

double getReticulationFirstParentProb(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    assert(
            network.edges_by_index[node->getReticulationData()->link_to_first_parent->edge_pmatrix_index]->prob
                    + network.edges_by_index[node->getReticulationData()->link_to_second_parent->edge_pmatrix_index]->prob
                    == 1.0);
    return network.edges_by_index[node->getReticulationData()->link_to_first_parent->edge_pmatrix_index]->prob;
}
double getReticulationSecondParentProb(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    assert(
            network.edges_by_index[node->getReticulationData()->link_to_first_parent->edge_pmatrix_index]->prob
                    + network.edges_by_index[node->getReticulationData()->link_to_second_parent->edge_pmatrix_index]->prob
                    == 1.0);
    return network.edges_by_index[node->getReticulationData()->link_to_second_parent->edge_pmatrix_index]->prob;
}

double getReticulationActiveProb(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    assert(
            network.edges_by_index[node->getReticulationData()->link_to_first_parent->edge_pmatrix_index]->prob
                    + network.edges_by_index[node->getReticulationData()->link_to_second_parent->edge_pmatrix_index]->prob
                    == 1.0);
    return network.edges_by_index[node->getReticulationData()->getLinkToActiveParent()->edge_pmatrix_index]->prob;
}

size_t getReticulationFirstParentPmatrixIndex(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToFirstParent()->edge_pmatrix_index;
}

size_t getReticulationSecondParentPmatrixIndex(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToSecondParent()->edge_pmatrix_index;
}

size_t getReticulationActiveParentPmatrixIndex(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToActiveParent()->edge_pmatrix_index;
}

std::vector<Node*> getChildren(Network &network, Node *node) {
    assert(node);
    std::vector<Node*> children;
    if (node->type == NodeType::RETICULATION_NODE) {
        children.push_back(getReticulationChild(network, node));
    } else { // normal node
        std::vector<Node*> neighbors = getNeighbors(network, node);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (getLinkToClvIndex(network, node, neighbors[i]->clv_index)->direction == Direction::OUTGOING) {
                children.push_back(neighbors[i]);
            }
        }
    }
    return children;
}

std::vector<Node*> getChildrenIgnoreDirections(Network &network, Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> children;
    if (node->type == NodeType::RETICULATION_NODE) {
        children.push_back(getReticulationChild(network, node));
    } else { // normal node
        std::vector<Node*> neighbors = getNeighbors(network, node);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (neighbors[i] != myParent) {
                children.push_back(neighbors[i]);
            }
        }
    }
    return children;
}

std::vector<Node*> getActiveChildren(Network &network, Node *node) {
    assert(node);
    std::vector<Node*> activeChildren;
    std::vector<Node*> children = getChildren(network, node);
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the child is active, this is, if we are currently the selected parent
            if (getActiveParent(network, children[i]) != node) {
                continue;
            }
        }
        activeChildren.push_back(children[i]);
    }
    assert(activeChildren.size() <= 2 || (node == network.root && activeChildren.size() == 3));
    return activeChildren;
}

std::vector<Node*> getActiveChildrenIgnoreDirections(Network &network, Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> activeChildren;
    std::vector<Node*> children = getChildrenIgnoreDirections(network, node, myParent);
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the child is active, this is, if we are currently the selected parent
            // or we could be the current child...
            if (node != getReticulationChild(network, children[i]) && getActiveParent(network, children[i]) != node) {
                continue;
            }
        }
        activeChildren.push_back(children[i]);
    }
    assert(activeChildren.size() <= 2 || (myParent == nullptr && activeChildren.size() == 3));
    return activeChildren;
}

Node* getOtherChild(Network &network, Node *parent, Node *aChild) {
    assert(parent && parent->type != NodeType::RETICULATION_NODE);
    assert(aChild);
    std::vector<Node*> children = getChildren(network, parent);
    assert(std::find(children.begin(), children.end(), aChild) != children.end());
    for (Node *child : children) {
        if (child != aChild) {
            return child;
        }
    }
    return nullptr;
}

bool hasChild(Network &network, Node *parent, Node *candidate) {
    assert(parent);
    assert(candidate);
    std::vector<Node*> children = getChildren(network, parent);
    return (std::find(children.begin(), children.end(), candidate) != children.end());
}

std::vector<Node*> getNeighbors(Network &network, const Node *node) {
    assert(node);
    std::vector<Node*> neighbors;
    for (const auto &link : node->links) {
        Node *target = getTargetNode(network, &link);
        neighbors.push_back(target);
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
        activeNeighbors.push_back(neighbors[i]);
    }
    assert(activeNeighbors.size() <= 3);
    return activeNeighbors;
}

Node* getActiveParent(Network &network, const Node *node) {
    assert(node);
    if (node->type == NodeType::RETICULATION_NODE) {
        return getReticulationActiveParent(network, node);
    }
    for (const auto &link : node->links) {
        if (link.direction == Direction::INCOMING) {
            return getTargetNode(network, &link);
        }
    }
    return nullptr;
}

std::vector<Node*> getAllParents(Network &network, const Node *node) {
    assert(node);
    std::vector<Node*> res;
    if (node->type == NodeType::RETICULATION_NODE) {
        res.emplace_back(getReticulationFirstParent(network, node));
        res.emplace_back(getReticulationSecondParent(network, node));
    } else {
        Node *parent = getActiveParent(network, node);
        if (parent) {
            res.emplace_back(parent);
        }
    }
    return res;
}

Edge* getEdgeTo(Network &network, const Node *node, const Node *target) {
    assert(node);
    assert(target);
    for (const auto &link : node->links) {
        if (link.outer->node_clv_index == target->clv_index) {
            return network.edges_by_index[link.edge_pmatrix_index];
        }
    }
    throw std::runtime_error("The given target node is not a neighbor of this node");
}

Edge* getEdgeTo(Network &network, size_t from_clv_index, size_t to_clv_index) {
    return getEdgeTo(network, network.nodes_by_index[from_clv_index], network.nodes_by_index[to_clv_index]);
}

std::vector<Edge*> getAdjacentEdges(Network &network, const Node *node) {
    std::vector<Edge*> res;
    std::vector<Node*> neighs = getNeighbors(network, node);
    for (size_t i = 0; i < neighs.size(); ++i) {
        res.emplace_back(getEdgeTo(network, node, neighs[i]));
    }
    assert(res.size() <= 3);
    return res;
}

Node* getSource(Network &network, const Edge *edge) {
    assert(edge->link1->direction == Direction::OUTGOING);
    return network.nodes_by_index[edge->link1->node_clv_index];
}

Node* getTarget(Network &network, const Edge *edge) {
    assert(edge->link2->direction == Direction::INCOMING);
    return network.nodes_by_index[edge->link2->node_clv_index];
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
    return false;
}

Link* make_link(Node *node, Edge *edge, Direction dir) {
    Link link;
    link.init(node ? node->clv_index : 0, edge ? edge->pmatrix_index : 0, nullptr, nullptr, dir);
    return node->addLink(link);
}

void invalidateHigherClvs(Network &network, pllmod_treeinfo_t *treeinfo, std::vector<bool> &visited, Node *node,
        bool invalidate_myself) {
    if (visited[node->clv_index]) {
        return;
    }
    if (invalidate_myself) {
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            treeinfo->clv_valid[p][node->clv_index] = 0;
        }
    }
    visited[node->clv_index] = true;
    if (node->clv_index == network.root->clv_index) {
        return;
    }
    if (node->type == NodeType::RETICULATION_NODE) {
        invalidateHigherClvs(network, treeinfo, visited, getReticulationFirstParent(network, node), true);
        invalidateHigherClvs(network, treeinfo, visited, getReticulationSecondParent(network, node), true);
    } else {
        invalidateHigherClvs(network, treeinfo, visited, getActiveParent(network, node), true);
    }
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, bool invalidate_myself) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    std::vector<bool> visited(ann_network.network.nodes.size(), false);
    invalidateHigherClvs(ann_network.network, treeinfo, visited, node, invalidate_myself);
}

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    for (size_t p = 0; p < treeinfo->partition_count; ++p) {
        treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
    }
}

}
