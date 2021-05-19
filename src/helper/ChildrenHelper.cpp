#include "Helper.hpp"
#include "../DebugPrintFunctions.hpp"

namespace netrax {

std::vector<Node*> getChildren(Network &network, const Node *node) {
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

std::vector<Node*> getChildrenIgnoreDirections(Network &network, const Node *node, const Node *myParent) {
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

std::vector<Node*> getActiveChildren(Network &network, const Node *node) {
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
        const Node *node) {
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

std::vector<Node*> getActiveChildrenUndirected(Network &network, const Node *node, const Node *myParent) {
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

Node* getOtherChild(Network &network, const Node *parent, const Node *aChild) {
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

bool hasChild(Network &network, const Node *parent, const Node *candidate) {
    assert(parent);
    assert(candidate);
    std::vector<Node*> children = getChildren(network, parent);
    return (std::find(children.begin(), children.end(), candidate) != children.end());
}

std::vector<Node*> getCurrentChildren(AnnotatedNetwork& ann_network, const Node* node, const Node* parent, const ReticulationConfigSet& restrictions) {
    assert(restrictions.configs.size() == 1);
    std::vector<Node*> children = getChildrenIgnoreDirections(ann_network.network, node, parent);
    std::vector<Node*> res;
    for (size_t i = 0; i < children.size(); ++i) {
        if (reticulationConfigsCompatible(restrictions, getRestrictionsToTakeNeighbor(ann_network, node, children[i]))) {
            res.emplace_back(children[i]);
        }
    }
    if (res.size() > 2) {
        std::cout << "Node: " << node->clv_index << "\n";
        if (parent) {
            std::cout << "Parent: " << parent->clv_index << "\n";
        } else {
            std::cout << "Parent: NULL" << "\n";
        }

        std::cout << exportDebugInfo(ann_network) << "\n";
    }
    assert(res.size() <= 2);
    return res;
}

}