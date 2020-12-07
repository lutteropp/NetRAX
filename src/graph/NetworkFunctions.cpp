/*
 * NetworkFunctions.cpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#include "NetworkFunctions.hpp"

#include <stdlib.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <queue>
#include <stack>
#include <stdexcept>

#include "BiconnectedComponents.hpp"
#include "Edge.hpp"
#include "Network.hpp"
#include "NetworkTopology.hpp"
#include "Node.hpp"
#include "NodeType.hpp"
#include "ReticulationData.hpp"
#include "AnnotatedNetwork.hpp"

namespace netrax {

static char* xstrdup(const char *s) {
    size_t len = strlen(s);
    char *p = (char*) malloc(len + 1);
    if (!p) {
        pll_errno = PLL_ERROR_MEM_ALLOC;
        snprintf(pll_errmsg, 200, "Memory allocation failed");
        return NULL;
    }
    return strcpy(p, s);
}

pll_unode_t* create_unode(const std::string &label) {
    pll_unode_t *new_unode = (pll_unode_t*) malloc(sizeof(pll_unode_t));
    if (!label.empty()) {
        new_unode->label = xstrdup(label.c_str());
    } else {
        new_unode->label = nullptr;
    }
    new_unode->scaler_index = -1;
    new_unode->clv_index = 0;
    new_unode->length = 0.0;
    new_unode->node_index = 0;
    new_unode->pmatrix_index = 0;
    new_unode->next = nullptr;
    return new_unode;
}

void destroy_unode(pll_unode_t *unode) {
    assert(unode);
    if (unode->label) {
        free(unode->label);
    }
    if (unode->next) {
        if (unode->next->next) {
            free(unode->next->next);
        }
        free(unode->next);
    }
    free(unode);
}

void remove_dead_children(std::vector<Node*> &children, const std::vector<bool> &dead_nodes) {
    children.erase(std::remove_if(children.begin(), children.end(), [&](Node *node) {
        assert(node);
        return (dead_nodes[node->clv_index]);
    }), children.end());
}

struct CumulatedChild {
    Node *child = nullptr;
    Node *direct_parent = nullptr;
    double cum_brlen = 0.0;
};

std::vector<Node*> getChildrenNoDir(Network &network, Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> children;
    if (node->type == NodeType::RETICULATION_NODE) {
        if (getReticulationChild(network, node) == myParent) {
            children.push_back(getActiveParent(network, node));
        } else {
            children.push_back(getReticulationChild(network, node));
        }
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

std::vector<Node*> getActiveChildrenNoDir(Network &network, Node *node, const Node *myParent) {
    assert(node);
    std::vector<Node*> activeChildren;
    std::vector<Node*> children = getChildrenNoDir(network, node, myParent);
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]->getType() == NodeType::RETICULATION_NODE) {
            // we need to check if the child is active, this is, if we are currently the selected parent
            if (((getReticulationFirstParent(network, children[i]) == node
                    || getReticulationSecondParent(network, children[i]) == node)
                    && getActiveParent(network, children[i]) != node)) {
                continue;
            }
        }
        activeChildren.push_back(children[i]);
    }
    assert(activeChildren.size() <= 2 || (myParent == nullptr && activeChildren.size() == 3));
    return activeChildren;
}

CumulatedChild getCumulatedChild(Network &network, Node *parent, Node *child,
        const std::vector<bool> &dead_nodes, const std::vector<bool> &skipped_nodes) {
    CumulatedChild res { child, parent, 0.0 };
    res.cum_brlen += getEdgeTo(network, child, parent)->length;
    Node *act_parent = parent;
    while (skipped_nodes[res.child->clv_index]) {
        std::vector<Node*> activeChildren = getActiveChildrenNoDir(network, res.child, act_parent);
        remove_dead_children(activeChildren, dead_nodes);
        assert(activeChildren.size() == 1);
        act_parent = res.child;
        res.child = activeChildren[0];
        res.cum_brlen += getEdgeTo(network, act_parent, res.child)->length;
    }
    res.direct_parent = act_parent;
    return res;
}

std::vector<CumulatedChild> getCumulatedChildren(Network &network, Node *parent, Node *actNode,
        const std::vector<bool> &dead_nodes, const std::vector<bool> &skipped_nodes) {
    assert(actNode);
    std::vector<CumulatedChild> res;
    assert(!skipped_nodes[actNode->clv_index]);
    std::vector<Node*> activeChildren = getActiveChildrenNoDir(network, actNode, parent);
    for (size_t i = 0; i < activeChildren.size(); ++i) {
        res.push_back(
                getCumulatedChild(network, actNode, activeChildren[i], dead_nodes, skipped_nodes));
    }
    return res;
}

pll_unode_t* connect_subtree_recursive(Network &network, Node *networkNode,
        pll_unode_t *from_parent, Node *networkParentNode, const std::vector<bool> &dead_nodes,
        const std::vector<bool> &skipped_nodes) {
    assert(networkNode->getType() == NodeType::BASIC_NODE);

    assert(!dead_nodes[networkNode->clv_index] && !skipped_nodes[networkNode->clv_index]);

    pll_unode_t *to_parent = nullptr;
    if (networkParentNode) {
        to_parent = create_unode(networkNode->getLabel());
        to_parent->clv_index = networkNode->clv_index;
        from_parent->back = to_parent;
        to_parent->back = from_parent;
        to_parent->length = from_parent->length;
        to_parent->next = NULL;
    }

    if (networkNode->isTip()) {
        return to_parent;
    }

    std::vector<CumulatedChild> cum_children = getCumulatedChildren(network, networkParentNode,
            networkNode, dead_nodes, skipped_nodes);
    assert(cum_children.size() == 2 || (cum_children.size() == 3 && networkParentNode == nullptr)); // 2 children, or started with root node.

    std::vector<pll_unode_t*> toChildren(cum_children.size(), nullptr);
    for (size_t i = 0; i < toChildren.size(); ++i) {
        toChildren[i] = create_unode(networkNode->getLabel());
        toChildren[i]->clv_index = networkNode->clv_index;
        toChildren[i]->length = cum_children[i].cum_brlen;
        connect_subtree_recursive(network, cum_children[i].child, toChildren[i],
                cum_children[i].direct_parent, dead_nodes, skipped_nodes);
    }

// set the next pointers
    bool isRoot = false;
    pll_unode_t *unode = to_parent;
    if (!unode) {
        unode = toChildren[0];
        isRoot = true;
    }

    for (size_t i = isRoot; i < toChildren.size(); ++i) {
        unode->next = toChildren[i];
        unode = unode->next;
    }

    unode->next = isRoot ? toChildren[0] : to_parent;

    return unode->next;
}

std::vector<bool> collect_dead_nodes(Network &network, size_t megablobRootClvIndex,
        Node **displayed_tree_root) {
    std::vector<bool> dead_nodes(network.nodes.size(), false);

    // seed the search with inactive reticulation parents
    std::queue<Node*> q;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        q.emplace(getReticulationNonActiveParent(network, network.reticulation_nodes[i]));
    }

    while (!q.empty()) {
        Node *u = q.front();
        q.pop();
        // if the node has no active and alive children, then it is dead.
        size_t num_undead = getActiveAliveChildren(network, dead_nodes, u).size();
        if (num_undead == 0) {
            dead_nodes[u->clv_index] = true;
            if (u->type == NodeType::RETICULATION_NODE) {
                q.emplace(getReticulationFirstParent(network, u));
                q.emplace(getReticulationSecondParent(network, u));
            } else {
                q.emplace(getActiveParent(network, u));
            }
        }
    }

    Node *dtroot = network.root;
    std::vector<Node*> children = getActiveAliveChildren(network, dead_nodes, dtroot);
    assert(!children.empty());
    bool seenMegablobRoot = false;
    while (children.size() == 1) {
        if (dtroot->clv_index == megablobRootClvIndex) {
            seenMegablobRoot = true;
        }
        dead_nodes[dtroot->clv_index] = true;
        dtroot = children[0];
        children = getActiveAliveChildren(network, dead_nodes, dtroot);
        assert(!children.empty());
    }
    if (displayed_tree_root) {
        if (seenMegablobRoot) {
            *displayed_tree_root = dtroot;
        } else {
            *displayed_tree_root = network.nodes_by_index[megablobRootClvIndex];
        }
    }

    return dead_nodes;
}

std::vector<bool> collect_skipped_nodes(Network &network, const std::vector<bool> &dead_nodes) {
    std::vector<bool> skipped_nodes(network.nodes.size(), false);

    for (size_t i = 0; i < network.num_nodes(); ++i) {
        if (network.nodes[i].isTip()) {
            continue; // tips are never skipped
        }
        skipped_nodes[network.nodes[i].clv_index] = (getActiveAliveNeighbors(network, dead_nodes,
                &network.nodes[i]).size() == 2);
        assert(
                dead_nodes[network.nodes[i].clv_index]
                        || getActiveAliveNeighbors(network, dead_nodes, &network.nodes[i]).size()
                                > 1);
    }

    return skipped_nodes;
}

double displayed_tree_prob(AnnotatedNetwork &ann_network, size_t tree_index) {
    setReticulationParents(ann_network.network, tree_index);
    double logProb = 0;
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        size_t active_pmatrix_idx = getReticulationActiveParentPmatrixIndex(ann_network.network.reticulation_nodes[i]);
        double prob = ann_network.branch_probs[active_pmatrix_idx];
        logProb += log(prob);
    }
    return exp(logProb);
}

pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index) {
    setReticulationParents(network, tree_index);

    std::vector<bool> dead_nodes = collect_dead_nodes(network, network.root->clv_index);
    Node *root = nullptr;

    root = getPossibleTreeRootNode(network, dead_nodes);
    assert(root);

    std::vector<bool> skipped_nodes = collect_skipped_nodes(network, dead_nodes);

    pll_unode_t *uroot = connect_subtree_recursive(network, root, nullptr, nullptr, dead_nodes,
            skipped_nodes);

    pll_utree_reset_template_indices(uroot, network.num_tips());
    pll_utree_t *utree = pll_utree_wraptree(uroot, network.num_tips());

// ensure that the tip clv indices are the same as in the network
    for (size_t i = 0; i < utree->inner_count + utree->tip_count; ++i) {
        if (utree->nodes[i]->clv_index < utree->tip_count) {
            Node *networkNode = network.getNodeByLabel(utree->nodes[i]->label);
            utree->nodes[i]->clv_index = utree->nodes[i]->node_index = networkNode->clv_index;
        }
    }

    assert(utree->tip_count == network.num_tips());
    return utree;
}

std::vector<double> collectBranchLengths(const Network &network) {
    std::vector<double> brLengths(network.num_branches());
    for (size_t i = 0; i < network.num_branches(); ++i) {
        brLengths[network.edges[i].pmatrix_index] = network.edges[i].length;
    }
    return brLengths;
}
void applyBranchLengths(Network &network, const std::vector<double> &branchLengths) {
    assert(branchLengths.size() == network.num_branches());
    for (size_t i = 0; i < network.num_branches(); ++i) {
        network.edges[i].length = branchLengths[network.edges[i].pmatrix_index];
    }
}
void setReticulationParents(Network &network, size_t treeIdx) {
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        // check if i-th bit is set in treeIdx
        bool activeParentIdx = treeIdx & (1 << i);
        network.reticulation_nodes[i]->getReticulationData()->setActiveParentToggle(
                activeParentIdx);
    }
}

void setReticulationParents(BlobInformation &blobInfo, unsigned int megablob_idx, size_t treeIdx) {
    for (size_t i = 0; i < blobInfo.reticulation_nodes_per_megablob[megablob_idx].size(); ++i) {
        // check if i-th bit is set in treeIdx
        bool activeParentIdx = treeIdx & (1 << i);
        blobInfo.reticulation_nodes_per_megablob[megablob_idx][i]->getReticulationData()->setActiveParentToggle(
                activeParentIdx);
    }
}

void forbidSubnetwork(Network &network, Node *node, std::vector<bool> &forbidden) {
    if (forbidden[node->clv_index])
        return;
    forbidden[node->clv_index] = true;
    std::vector<Node*> children = getChildren(network, node);
    for (size_t i = 0; i < children.size(); ++i) {
        forbidSubnetwork(network, children[i], forbidden);
    }
}

/*
 * Find possible placements for the root node in a semi-rooted network.
 */
std::vector<Node*> getPossibleRootNodes(Network &network) {
    std::vector<bool> forbidden(network.num_nodes(), false);
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        forbidSubnetwork(network, network.reticulation_nodes[i], forbidden);
    }
    std::vector<Node*> res;
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        if (!forbidden[network.nodes[i].clv_index]) {
            if (network.nodes[i].getType() == NodeType::BASIC_NODE
                    && getActiveNeighbors(network, &network.nodes[i]).size() == 3) {
                res.push_back(&network.nodes[i]);
            }
        }
    }
    return res;
}

Node* getPossibleTreeRootNode(Network &network, const std::vector<bool> &dead_nodes) {
    if (!dead_nodes[network.root->clv_index]
            && getActiveAliveNeighbors(network, dead_nodes, network.root).size() == 3) {
        return network.root;
    }
    for (size_t i = network.num_tips(); i < network.num_nodes(); ++i) {
        if (!dead_nodes[network.nodes[i].clv_index]
                && network.nodes[i].getType() == NodeType::BASIC_NODE
                && getActiveAliveNeighbors(network, dead_nodes, &network.nodes[i]).size() == 3) {
            return &network.nodes[i];
        }
    }
    return nullptr;
}

void getTipVectorRecursive(pll_unode_t *actParent, pll_unode_t *actNode, size_t pmatrix_idx,
        bool pmatrix_idx_found, std::vector<bool> &res) {
    if (!actNode) {
        return;
    }
    if (actNode->pmatrix_index == pmatrix_idx) {
        pmatrix_idx_found = true;
    }
    if (pllmod_utree_is_tip(actNode) && pmatrix_idx_found) {
        res[actNode->clv_index] = true;
    } else if (!pllmod_utree_is_tip(actNode)) {

        pll_unode_t *link = actNode->next;
        do {
            if (link->back != actParent) {
                getTipVectorRecursive(actNode, link->back, pmatrix_idx, pmatrix_idx_found, res);
            }
            link = link->next;
        } while (link && link != actNode);
    }
}

std::vector<bool> getTipVector(const pll_utree_t &utree, size_t pmatrix_idx) {
    std::vector<bool> res(utree.tip_count, false);
// do a top-down preorder traversal of the tree,
//	starting to write to the tip vector as soon as we have encountered the wanted pmatrix_idx
    getTipVectorRecursive(utree.vroot->back, utree.vroot, pmatrix_idx, false, res);

// vroot and vroot->back have the same pmatrix index!!!
    if (utree.vroot->pmatrix_index != pmatrix_idx) {
        getTipVectorRecursive(utree.vroot, utree.vroot->back, pmatrix_idx, false, res);
    }
    return res;
}

void getTipVectorRecursive(Network &network, Node *actParent, Node *actNode, size_t pmatrix_idx,
        bool pmatrix_idx_found, std::vector<bool> &res) {
    if ((actParent != nullptr)
            && (getEdgeTo(network, actNode, actParent)->pmatrix_index == pmatrix_idx)) {
        pmatrix_idx_found = true;
    }
    if (actNode->isTip() && pmatrix_idx_found) {
        res[actNode->clv_index] = true;
    } else if (!actNode->isTip()) {
        std::vector<Node*> activeChildren = getActiveChildrenUndirected(network, actNode,
                actParent);
        for (size_t i = 0; i < activeChildren.size(); ++i) {
            getTipVectorRecursive(network, actNode, activeChildren[i], pmatrix_idx,
                    pmatrix_idx_found, res);
        }
    }
}

std::vector<bool> getTipVector(Network &network, size_t pmatrix_idx) {
    std::vector<bool> res(network.num_tips(), false);
// do a top-down preorder traversal of the network,
//	starting to write to the tip vector as soon as we have encountered the wanted pmatrix_idx
    getTipVectorRecursive(network, nullptr, network.root, pmatrix_idx, false, res);
    return res;
}

std::vector<std::vector<size_t> > getDtBranchToNetworkBranchMapping(const pll_utree_t &utree,
        Network &network, size_t tree_idx) {
    std::vector<std::vector<size_t> > res(utree.edge_count);
    setReticulationParents(network, tree_idx);

// for each branch, we need to figure out which tips are on one side of the branch, and which tips are on the other side
// so essentially, we need to compare bipartitions. That's all!

// ... we can easily get the set of tips which are in a subtree!
//  (of either of the endpoints of the current branch, we don't really care)!!!

// and we can use a bool vector for all tips...

    std::vector<std::vector<bool> > networkTipVectors(network.num_branches());
    for (size_t i = 0; i < network.num_branches(); ++i) {
        networkTipVectors[i] = getTipVector(network, i);
    }

    for (size_t i = 0; i < utree.edge_count; ++i) {
        std::vector<bool> tipVecTree = getTipVector(utree, i);
        for (size_t j = 0; j < network.num_branches(); ++j) {
            std::vector<bool> tipVecNetwork = networkTipVectors[j];
            if (tipVecTree == tipVecNetwork) {
                res[i].push_back(j);
            } else {
                // check if they are all different
                bool allDifferent = true;
                for (size_t k = 0; k < tipVecTree.size(); ++k) {
                    if (tipVecTree[k] == tipVecNetwork[k]) {
                        allDifferent = false;
                    }
                }
                if (allDifferent) {
                    res[i].push_back(j);
                }
            }
        }
    }
    return res;
}

std::vector<Node*> grab_current_node_parents(Network &network) {
    std::vector<Node*> parent(network.nodes.size(), nullptr);
    for (size_t i = 0; i < parent.size(); ++i) {
        if (network.nodes_by_index[i]) {
            parent[network.nodes_by_index[i]->clv_index] = getActiveParent(network,
                    network.nodes_by_index[i]);
        }
    }
    return parent;
}

std::vector<Node*> reversed_topological_sort(Network &network) {
    std::vector<Node*> res;
    res.reserve(network.num_nodes());
    std::vector<Node*> parent = grab_current_node_parents(network);
    std::vector<unsigned int> outdeg(network.nodes.size(), 0);

    std::queue<Node*> q;

// Kahn's algorithm for topological sorting

// compute outdegree of all nodes
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        Node *actNode = &network.nodes[i];
        size_t act_clv_idx = actNode->clv_index;
        outdeg[act_clv_idx] = getChildren(network, actNode).size();
        if (outdeg[act_clv_idx] == 0) {
            q.emplace(actNode);
        }
    }

//std::cout << exportDebugInfo(network, outdeg) << "\n";

    size_t num_visited_vertices = 0;
    while (!q.empty()) {
        Node *actNode = q.front();
        q.pop();
        res.emplace_back(actNode);

        if (actNode->type == NodeType::BASIC_NODE) {
            if (parent[actNode->clv_index] != nullptr) { // catch special case for root node
                outdeg[parent[actNode->clv_index]->clv_index]--;
                if (outdeg[parent[actNode->clv_index]->clv_index] == 0) {
                    q.emplace(parent[actNode->clv_index]);
                }
            }
        } else { // reticulation node. It has 2 parents and 1 child
            for (Node *neigh : getNeighbors(network, actNode)) {
                if (parent[neigh->clv_index] != actNode) {
                    outdeg[neigh->clv_index]--;
                    if (outdeg[neigh->clv_index] == 0) {
                        q.emplace(neigh);
                    }
                }
            }
        }
        num_visited_vertices++;
    }

    if (num_visited_vertices != network.num_nodes()) {
        throw std::runtime_error("Cycle in network detected");
    }

    return res;
}

bool networkIsConnected(Network &network) {
    unsigned int n_visited = 0;
    std::vector<bool> visited(network.num_nodes(), false);
    std::stack<const Node*> s;
    s.emplace(network.root);
    while (!s.empty()) {
        const Node *actNode = s.top();
        s.pop();
        if (visited[actNode->clv_index]) {
            continue;
        }
        visited[actNode->clv_index] = true;
        n_visited++;
        for (const Node *neigh : getNeighbors(network, actNode)) {
            if (!visited[neigh->clv_index]) {
                s.emplace(neigh);
            }
        }
    }
    return (n_visited == network.num_nodes());
}

}
