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
#include "../DebugPrintFunctions.hpp"

namespace netrax {

Node* getTargetNode(const Network &network, const Link *link) {
    assert(link);
    if (network.edges_by_index[link->edge_pmatrix_index]->link1 == link) {
        return network.nodes_by_index[network.edges_by_index[link->edge_pmatrix_index]->link2->node_clv_index];
    } else {
        assert(network.edges_by_index[link->edge_pmatrix_index]->link2 == link);
        return network.nodes_by_index[network.edges_by_index[link->edge_pmatrix_index]->link1->node_clv_index];
    }
}

bool isOutgoing(Network &network, Node *from, Node *to) {
    assert(!getLinksToClvIndex(network, from, to->clv_index).empty());
    auto children = getChildren(network, from);
    return (std::find(children.begin(), children.end(), to) != children.end());
}

std::vector<Link*> getLinksToClvIndex(Network &network, Node *node, size_t target_index) {
    assert(node);
    std::vector<Link*> res;
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(network, &(node->links[i]))->clv_index == target_index) {
            res.emplace_back(&(node->links[i]));
        }
    }
    return res;
}

Link* getLinkToNode(Network &network, Node *node, Node *target) {
    assert(node);
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(network, &(node->links[i])) == target) {
            return &(node->links[i]);
        }
    }
    throw std::runtime_error("the node is not a neighbor");
}

Link* getLinkToNode(Network &network, size_t from_clv_index, size_t to_clv_index) {
    return getLinkToNode(network, network.nodes_by_index[from_clv_index],
            network.nodes_by_index[to_clv_index]);
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

Node* getReticulationNonActiveParent(Network &network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return getTargetNode(network, node->getReticulationData()->getLinkToNonActiveParent());
}

double getReticulationFirstParentProb(AnnotatedNetwork &ann_network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return ann_network.reticulation_probs[node->getReticulationData()->reticulation_index];
}

double getReticulationSecondParentProb(AnnotatedNetwork &ann_network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return 1.0 - ann_network.reticulation_probs[node->getReticulationData()->reticulation_index];
}

double getReticulationActiveProb(AnnotatedNetwork &ann_network, const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    size_t first_parent_pmatrix_index = getReticulationFirstParentPmatrixIndex(node);
    size_t active_pmatrix_index =
            node->getReticulationData()->getLinkToActiveParent()->edge_pmatrix_index;

    if (first_parent_pmatrix_index == active_pmatrix_index) {
        return getReticulationFirstParentProb(ann_network, node);
    } else {
        return getReticulationSecondParentProb(ann_network, node);
    }
}

size_t getReticulationFirstParentPmatrixIndex(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToFirstParent()->edge_pmatrix_index;
}

size_t getReticulationSecondParentPmatrixIndex(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToSecondParent()->edge_pmatrix_index;
}

size_t getReticulationActiveParentPmatrixIndex(const Node *node) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    return node->getReticulationData()->getLinkToActiveParent()->edge_pmatrix_index;
}

Node* getReticulationOtherParent(Network &network, const Node *node, const Node *parent) {
    assert(node);
    assert(node->type == NodeType::RETICULATION_NODE);
    if (getReticulationFirstParent(network, node) == parent) {
        return getReticulationSecondParent(network, node);
    } else {
        assert(getReticulationSecondParent(network, node) == parent);
        return getReticulationFirstParent(network, node);
    }
}

std::vector<Node*> getChildren(Network &network, Node *node) {
    assert(node);
    std::vector<Node*> children;
    if (node->type == NodeType::RETICULATION_NODE) {
        children.push_back(getReticulationChild(network, node));
    } else { // normal node
        std::vector<Node*> neighbors = getNeighbors(network, node);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (getLinksToClvIndex(network, node, neighbors[i]->clv_index)[0]->direction
                    == Direction::OUTGOING) {
                children.push_back(neighbors[i]);
            }
        }
    }
    return children;
}

std::vector<Node*> getChildrenIgnoreDirections(Network &network, Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> children;
    
    std::vector<Node*> neighbors = getNeighbors(network, node);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        if (neighbors[i] != myParent) {
            children.push_back(neighbors[i]);
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

std::vector<Node*> getActiveAliveChildren(Network &network, const std::vector<bool> &dead_nodes,
        Node *node) {
    assert(node);
    std::vector<Node*> activeChildren;
    std::vector<Node*> children = getChildren(network, node);
    for (size_t i = 0; i < children.size(); ++i) {
        if (dead_nodes[children[i]->clv_index]) {
            continue;
        }
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

std::vector<Node*> getActiveChildrenUndirected(Network &network, Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> activeChildren;
    std::vector<Node*> children = getChildrenIgnoreDirections(network, node, myParent);
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the child is active, this is, if we are currently the selected parent
            // or we could be the current child...
            if (node != getReticulationChild(network, children[i])
                    && getActiveParent(network, children[i]) != node) {
                continue;
            }
        }
        activeChildren.push_back(children[i]);
    }
    assert(activeChildren.size() <= 2 || (node == network.root && activeChildren.size() == 3));
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
    return getEdgeTo(network, network.nodes_by_index[from_clv_index],
            network.nodes_by_index[to_clv_index]);
}

std::vector<Edge*> getAdjacentEdges(Network &network, const Node *node) {
    std::vector<Edge*> res;
    std::vector<Node*> neighs = getNeighbors(network, node);
    for (size_t i = 0; i < neighs.size(); ++i) {
        res.emplace_back(getEdgeTo(network, node, neighs[i]));
    }
    assert(res.size() == 1 || res.size() == 3 || (node == network.root && res.size() == 2));
    return res;
}

std::vector<Edge*> getAdjacentEdges(Network &network, const Edge *edge) {
    std::vector<Edge*> res;

    Node *node1 = network.nodes_by_index[edge->link1->node_clv_index];
    Node *node2 = network.nodes_by_index[edge->link2->node_clv_index];

    for (size_t i = 0; i < node1->links.size(); ++i) {
        if (node1->links[i].edge_pmatrix_index != edge->pmatrix_index) {
            Edge *neigh = network.edges_by_index[node1->links[i].edge_pmatrix_index];
            if (std::find(res.begin(), res.end(), neigh) == res.end()) {
                res.emplace_back(neigh);
            }
        }
    }

    for (size_t i = 0; i < node2->links.size(); ++i) {
        if (node2->links[i].edge_pmatrix_index != edge->pmatrix_index) {
            Edge *neigh = network.edges_by_index[node2->links[i].edge_pmatrix_index];
            if (std::find(res.begin(), res.end(), neigh) == res.end()) {
                res.emplace_back(neigh);
            }
        }
    }
    assert(
            res.size() == 2 || res.size() == 4 || (node1 == network.root && res.size() == 3)
                    || (node1 == network.root && node2->isTip() && res.size() == 1));
    return res;
}

Node* getSource(Network &network, const Edge *edge) {
    assert(edge);
    assert(edge->link1);
    assert(edge->link2);
    assert(edge->link1->direction == Direction::OUTGOING);
    return network.nodes_by_index[edge->link1->node_clv_index];
}

Node* getTarget(Network &network, const Edge *edge) {
    assert(edge);
    assert(edge->link1);
    assert(edge->link2);
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

    for (const auto &link : node2->links) {
        if (link.outer->node_clv_index == node1->clv_index) {
            throw std::runtime_error("The links are not symmetric");
        }
    }
    return false;
}

Link* make_link(Node *node, Edge *edge, Direction dir) {
    Link link;
    link.init(node ? node->clv_index : 0, edge ? edge->pmatrix_index : 0, nullptr, nullptr, dir);
    return node->addLink(link);
}

void invalidateSingleClv(pllmod_treeinfo_t *treeinfo, unsigned int clv_index) {
    for (size_t p = 0; p < treeinfo->partition_count; ++p) {
        treeinfo->clv_valid[p][clv_index] = 0;
    }
}

void invalidateHigherClvs(AnnotatedNetwork &ann_network, pllmod_treeinfo_t *treeinfo, Node *node, size_t partition_idx,
        bool invalidate_myself, std::vector<bool> &visited) {
    Network &network = ann_network.network;
    if (!node) {
        return;
    }
    if (!visited.empty() && visited[node->clv_index]) { // clv at node is already invalidated
        return;
    }
    if (invalidate_myself) {
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            treeinfo->clv_valid[p][node->clv_index] = 0;
        }
        if (!visited.empty()) {
            visited[node->clv_index] = true;
        }
    }
    if (node->clv_index == network.root->clv_index) {
        return;
    }
    if (node->type == NodeType::RETICULATION_NODE) {
        invalidateHigherClvs(ann_network, treeinfo, getReticulationFirstParent(network, node), partition_idx, true,
                visited);
        invalidateHigherClvs(ann_network, treeinfo, getReticulationSecondParent(network, node), partition_idx,
                true, visited);
    } else {
        invalidateHigherClvs(ann_network, treeinfo, getActiveParent(network, node), partition_idx, true, visited);
    }
    ann_network.cached_logl_valid = false;
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, size_t partition_idx, bool invalidate_myself,
        std::vector<bool> &visited) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    invalidateHigherClvs(ann_network, treeinfo, node, partition_idx, invalidate_myself, visited);
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, bool invalidate_myself, std::vector<bool> &visited) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        invalidateHigherCLVs(ann_network, node, p, invalidate_myself, visited);
    }
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, Node *node, size_t partition_idx, bool invalidate_myself) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    std::vector<bool> noVisited;
    invalidateHigherClvs(ann_network, treeinfo, node, partition_idx, invalidate_myself, noVisited);
}

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index,
        std::vector<bool> &visited) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    for (size_t p = 0; p < treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        invalidateHigherCLVs(ann_network,
            getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]), p, true,
            visited);
    }
}

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index) {
    std::vector<bool> noVisited;
    invalidatePmatrixIndex(ann_network, pmatrix_index, noVisited);
}

void invalidPmatrixIndexOnly(AnnotatedNetwork& ann_network, size_t pmatrix_index) {
    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
            continue;
        }
        ann_network.fake_treeinfo->pmatrix_valid[partition_idx][pmatrix_index] = 0;
    }
    ann_network.cached_logl_valid = false;
}

bool assertReticulationProbs(AnnotatedNetwork &ann_network) {
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t n_partitions = 1;
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    for (size_t p = 0; p < n_partitions; ++p) {
        for (size_t i = 0; i < ann_network.network.reticulation_nodes.size(); ++i) {
            double actProb = getReticulationActiveProb(ann_network,
                    ann_network.network.reticulation_nodes[i]);
            assert(actProb >= 0 && actProb <= 1);
        }
    }
    return true;
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

void setReticulationState(Network &network, size_t reticulation_idx, ReticulationState state) {
    assert(reticulation_idx < network.reticulation_nodes.size());
    if (state == ReticulationState::DONT_CARE) {
        return;
    } else if (state == ReticulationState::TAKE_FIRST_PARENT) {
        network.reticulation_nodes[reticulation_idx]->getReticulationData()->setActiveParentToggle(0);
    } else { // TAKE_SECOND_PARENT
        network.reticulation_nodes[reticulation_idx]->getReticulationData()->setActiveParentToggle(1);
    }
}

}
