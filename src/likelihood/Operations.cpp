/*
 * Operations.cpp
 *
 *  Created on: Jun 4, 2020
 *      Author: sarah
 */

#include "Operations.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <utility>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}
#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/Edge.hpp"
#include "../graph/Network.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../graph/NodeType.hpp"
#include "../DebugPrintFunctions.hpp"

namespace netrax {
pll_operation_t buildOperationInternal(Network &network, Node *parent, Node *child1, Node *child2,
        size_t fake_clv_index, size_t fake_pmatrix_index) {
    pll_operation_t operation;
    assert(parent);
    operation.parent_clv_index = parent->clv_index;
    operation.parent_scaler_index = parent->scaler_index;
    if (child1) {
        operation.child1_clv_index = child1->clv_index;
        operation.child1_scaler_index = child1->scaler_index;
        operation.child1_matrix_index = getEdgeTo(network, child1, parent)->pmatrix_index;
    } else {
        operation.child1_clv_index = fake_clv_index;
        operation.child1_scaler_index = -1;
        operation.child1_matrix_index = fake_pmatrix_index;
    }
    if (child2) {
        operation.child2_clv_index = child2->clv_index;
        operation.child2_scaler_index = child2->scaler_index;
        operation.child2_matrix_index = getEdgeTo(network, child2, parent)->pmatrix_index;
    } else {
        operation.child2_clv_index = fake_clv_index;
        operation.child2_scaler_index = -1;
        operation.child2_matrix_index = fake_pmatrix_index;
    }
    return operation;
}

pll_operation_t buildOperation(Network &network, Node *actNode, Node *actParent,
        const std::vector<bool> &dead_nodes, size_t fake_clv_index, size_t fake_pmatrix_index) {
    std::vector<Node*> activeChildren = getActiveChildrenUndirected(network, actNode, actParent);
    assert(activeChildren.size() > 0);
    Node *child1 = nullptr;
    if (!dead_nodes[activeChildren[0]->clv_index]) {
        child1 = activeChildren[0];
    }
    Node *child2 = nullptr;
    if (activeChildren.size() == 2 && !dead_nodes[activeChildren[1]->clv_index]) {
        child2 = activeChildren[1];
    }
    return buildOperationInternal(network, actNode, child1, child2, fake_clv_index,
            fake_pmatrix_index);
}

void createOperationsPostorder(AnnotatedNetwork &ann_network, bool incremental,
        size_t partition_idx, Node *actNode, Node *parent, std::vector<pll_operation_t> &ops,
        size_t fake_clv_index, size_t fake_pmatrix_index, const std::vector<bool> &dead_nodes,
        const std::vector<unsigned int> *stop_indices) {
    if (stop_indices
            && std::find(stop_indices->begin(), stop_indices->end(), actNode->clv_index)
                    != stop_indices->end()) {
        return;
    }
    Network &network = ann_network.network;
    if (incremental && ann_network.fake_treeinfo->clv_valid[partition_idx][actNode->clv_index]) {
        return;
    }

    std::vector<Node*> activeChildren = getActiveChildrenUndirected(network, actNode, parent);
    if (activeChildren.empty()) {
        return;
    }
    assert(activeChildren.size() <= 2);
    for (size_t i = 0; i < activeChildren.size(); ++i) {
        createOperationsPostorder(ann_network, incremental, partition_idx, activeChildren[i],
                actNode, ops, fake_clv_index, fake_pmatrix_index, dead_nodes, stop_indices);
    }

    pll_operation_t operation = buildOperation(network, actNode, parent, dead_nodes, fake_clv_index,
            fake_pmatrix_index);
    ops.push_back(operation);
}

void fill_untouched_ops_recursive(Network &network, std::vector<bool> &clv_touched,
        const std::vector<bool> &dead_nodes, std::vector<pll_operation_t> &ops, Node *node) {
    if (clv_touched[node->clv_index]) {
        return;
    }
    std::vector<Node*> children = getActiveAliveChildren(network, dead_nodes, node);
    assert(!children.empty());
    for (size_t i = 0; i < children.size(); ++i) {
        fill_untouched_ops_recursive(network, clv_touched, dead_nodes, ops, children[i]);
    }
    ops.emplace_back(
            buildOperation(network, node, getActiveParent(network, node), dead_nodes,
                    network.nodes.size(), network.edges.size()));

    clv_touched[node->clv_index] = true;
}

void fill_untouched_clvs(AnnotatedNetwork &ann_network, std::vector<bool> &clv_touched,
        const std::vector<bool> &dead_nodes, size_t partition_idx, Node *startNode) {
    std::vector<pll_operation_t> ops;
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    fill_untouched_ops_recursive(network, clv_touched, dead_nodes, ops, startNode);
    size_t ops_count = ops.size();
    pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);
}

Node* getActiveParent(Network &network, Node *actNode, const std::vector<Node*> &parent) {
    Node *actParent;
    if (actNode->type == NodeType::RETICULATION_NODE) {
        actParent = getReticulationActiveParent(network, actNode);
    } else {
        actParent = parent[actNode->clv_index];
    }
    return actParent;
}

// forward declaration needed for createOperationsTowardsRoot
std::vector<pll_operation_t> createOperationsUpdatedReticulation(AnnotatedNetwork &ann_network,
        size_t partition_idx, const std::vector<Node*> &parent, Node *actNode,
        const std::vector<bool> &dead_nodes, bool incremental, Node *displayed_tree_root);

bool assertNoDuplicateOperations(const std::vector<pll_operation_t> &ops) {
    for (size_t i = 0; i < ops.size(); ++i) {
        for (size_t j = i + 1; j < ops.size(); ++j) {
            assert(
                    ops[i].parent_clv_index != ops[j].parent_clv_index
                            || ops[i].child1_clv_index != ops[j].child1_clv_index
                            || ops[i].child2_clv_index != ops[j].child2_clv_index);
        }
    }
    return true;
}

std::vector<pll_operation_t> createOperationsTowardsRoot(AnnotatedNetwork &ann_network,
        size_t partition_idx, const std::vector<Node*> &parent, Node *actParent,
        const std::vector<bool> &dead_nodes, bool incremental, Node *displayed_tree_root) {
    Network &network = ann_network.network;

    std::vector<pll_operation_t> ops;
    if (actParent == parent[displayed_tree_root->clv_index]) {
        return ops;
    }
    if (dead_nodes[actParent->clv_index]) {
        if (actParent == network.root) {
            return ops;
        } else if (actParent->type == NodeType::RETICULATION_NODE) {
            // when our current node is a dead reticulation node, we need to again go up from both
            // its parents, not just from the active parent...
            return createOperationsUpdatedReticulation(ann_network, partition_idx, parent,
                    actParent, dead_nodes, incremental, displayed_tree_root);
        } else {
            return createOperationsTowardsRoot(ann_network, partition_idx, parent,
                    parent[actParent->clv_index], dead_nodes, incremental, displayed_tree_root);
        }
    }

    size_t fake_clv_index = network.nodes.size();
    size_t fake_pmatrix_index = network.edges.size();

    Node *lastParent = network.root;
    if (displayed_tree_root != network.root) {
        lastParent = getActiveParent(network, displayed_tree_root);
    }

    while (actParent != lastParent) {
        if (!incremental || ann_network.network.num_reticulations() != 0
                || !ann_network.fake_treeinfo->clv_valid[partition_idx][actParent->clv_index]) {
            ops.emplace_back(
                    buildOperation(network, actParent, parent[actParent->clv_index], dead_nodes,
                            fake_clv_index, fake_pmatrix_index));
        }
        actParent = getActiveParent(network, actParent, parent);
    }

    bool toplevel_trifurcation = (getChildren(network, network.root).size() == 3);
    assert(!toplevel_trifurcation);
    if (displayed_tree_root == network.root && toplevel_trifurcation) {
        Node *rootBack = getTargetNode(network, network.root->getLink());
        if (!getActiveChildrenUndirected(network, network.root, rootBack).empty()) {
            ops.push_back(
                    buildOperation(network, network.root, rootBack, dead_nodes, fake_clv_index,
                            fake_pmatrix_index));
        } else {
            // special case: the root has a single child.
            ops.push_back(
                    buildOperationInternal(network, network.root, rootBack, nullptr, fake_clv_index,
                            fake_pmatrix_index));
            // ignore the branch length from the root to its single active child/ treat it
            // as if it had zero branch lBenoitength
            ops[ops.size() - 1].child1_matrix_index = fake_pmatrix_index;
        }
    } // else since displayed tree root has only 2 children, no need for rootBack stuff
    else if (displayed_tree_root == network.root) {
        ops.push_back(
                    buildOperation(network, network.root, nullptr, dead_nodes, fake_clv_index,
                            fake_pmatrix_index));
    }

    assert(assertNoDuplicateOperations(ops));

    // just for debug
    /*std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << "operations array:\n";
    printOperationArray(ops);
    std::cout << "\n";
    std::cout << "displayed tree root clv index: " << displayed_tree_root->clv_index << "\n";*/

    assert(ops[ops.size() - 1].parent_clv_index == displayed_tree_root->clv_index);
    return ops;
}

std::vector<pll_operation_t> createOperationsUpdatedReticulation(AnnotatedNetwork &ann_network,
        size_t partition_idx, const std::vector<Node*> &parent, Node *actNode,
        const std::vector<bool> &dead_nodes, bool incremental, Node *displayed_tree_root) {
    Network &network = ann_network.network;
    std::vector<pll_operation_t> ops;

    Node *firstParent = getReticulationFirstParent(network, actNode);
    std::vector<pll_operation_t> opsFirst = createOperationsTowardsRoot(ann_network, partition_idx,
            parent, firstParent, dead_nodes, incremental, displayed_tree_root);
    Node *secondParent = getReticulationSecondParent(network, actNode);
    std::vector<pll_operation_t> opsSecond = createOperationsTowardsRoot(ann_network, partition_idx,
            parent, secondParent, dead_nodes, incremental, displayed_tree_root);

    // find the first entry in opsFirst which also occurs in opsSecond.
    // We will only take opsFirst until this entry, excluding it.
    std::unordered_set<unsigned int> opsSecondRoots;
    for (size_t i = 0; i < opsSecond.size(); ++i) {
        opsSecondRoots.emplace(opsSecond[i].parent_clv_index);
    }
    for (size_t i = 0; i < opsFirst.size(); ++i) {
        if (opsSecondRoots.find(opsFirst[i].parent_clv_index) != opsSecondRoots.end()) {
            break;
        }
        ops.emplace_back(opsFirst[i]);
    }
    for (size_t i = 0; i < opsSecond.size(); ++i) {
        ops.emplace_back(opsSecond[i]);
    }

    assert(ops.empty() || ops[ops.size() - 1].parent_clv_index == displayed_tree_root->clv_index);
    return ops;
}

}
