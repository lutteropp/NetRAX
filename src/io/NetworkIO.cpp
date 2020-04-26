/*
 * NetworkConverter.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "NetworkIO.hpp"

#include <stddef.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stack>

#include "../graph/Direction.hpp"
#include "../graph/Edge.hpp"
#include "../graph/Link.hpp"
#include "../graph/Network.hpp"
#include "../graph/Node.hpp"
#include "../graph/NodeType.hpp"
#include "../graph/ReticulationData.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../graph/NetworkTopology.hpp"

namespace netrax {

std::vector<RootedNetworkNode*> collectNodes(RootedNetwork &rnetwork) {
    std::vector<RootedNetworkNode*> res;
    std::stack<RootedNetworkNode*> s;
    std::unordered_set<RootedNetworkNode*> visited;
    s.emplace(rnetwork.root);
    while (!s.empty()) {
        RootedNetworkNode *actNode = s.top();
        s.pop();
        if (visited.find(actNode) != visited.end()) {
            continue;
        }
        visited.emplace(actNode);
        res.emplace_back(actNode);
        for (RootedNetworkNode *child : actNode->children) {
            if (visited.find(child) == visited.end()) {
                s.emplace(child);
            }
        }
    }
    return res;
}

Network convertNetworkToplevelTrifurcation(RootedNetwork &rnetwork, size_t node_count, size_t branch_count,
        int maxReticulations) {
    Network network;

    network.branchCount = branch_count;
    network.nodeCount = node_count;
    network.tipCount = rnetwork.tipCount;

    if (maxReticulations >= 0 && (unsigned int) maxReticulations < rnetwork.reticulationCount) {
        throw std::runtime_error("number of reticulations in the network is higher than maximum reticulation count");
    }
    unsigned int maxExtraReticulations = (maxReticulations != -1) ? maxReticulations - rnetwork.reticulationCount : 0;
    unsigned int maxNodeCount = node_count + maxExtraReticulations * 2;
    unsigned int maxBranchCount = branch_count + maxExtraReticulations * 3;

    network.nodes.resize(maxNodeCount);
    network.edges.resize(maxBranchCount);
    // initialize the access by clv_index and pmatrix_index
    network.nodes_by_index.resize(node_count);
    for (size_t i = 0; i < maxNodeCount; ++i) {
        network.nodes_by_index[i] = &network.nodes[i];
    }
    network.edges_by_index.resize(branch_count);
    for (size_t i = 0; i < maxBranchCount; ++i) {
        network.edges_by_index[i] = &network.edges[i];
    }
    network.reticulation_nodes.resize(rnetwork.reticulationCount);

    std::vector<RootedNetworkNode*> rnetwork_nodes = collectNodes(rnetwork);
    assert(rnetwork_nodes.size() == node_count);
    std::vector<RootedNetworkNode*> rnetwork_tips, rnetwork_inner_tree, rnetwork_reticulations;
    for (RootedNetworkNode *ptr : rnetwork_nodes) {
        if (ptr->children.empty()) {
            rnetwork_tips.emplace_back(ptr);
        } else {
            if (ptr->isReticulation) {
                rnetwork_reticulations.emplace_back(ptr);
            } else {
                if (ptr != rnetwork.root) {
                    rnetwork_inner_tree.emplace_back(ptr);
                }
            }
        }
    }
    rnetwork_inner_tree.emplace_back(rnetwork.root);

    // 1.) Create all the nodes and edges. Also create all the incoming links.
    for (RootedNetworkNode *rnode : rnetwork_tips) {
        size_t clv_index = rnode->tip_index;
        rnode->clv_index = clv_index;
        int scaler_index = -1;
        assert(clv_index < network.num_nodes());
        network.nodes[clv_index].initBasic(clv_index, scaler_index, rnode->label);
        size_t pmatrix_index = clv_index;
        assert(pmatrix_index < network.num_branches());
        network.edges[pmatrix_index].init(pmatrix_index, nullptr, nullptr, rnode->length);

        Link *linkToParent = make_link(&network.nodes[clv_index], &network.edges[pmatrix_index],
                Direction::INCOMING);
        network.edges[pmatrix_index].link1 = linkToParent;
    }

    for (size_t i = 0; i < rnetwork_inner_tree.size(); ++i) {
        RootedNetworkNode *rnode = rnetwork_inner_tree[i];
        size_t clv_index = i + rnetwork_tips.size();
        rnode->clv_index = clv_index;
        int scaler_index = i;

        assert(clv_index < network.num_nodes());
        network.nodes[clv_index].initBasic(clv_index, scaler_index, rnode->label);

        if (rnode != rnetwork.root) {
            size_t pmatrix_index = clv_index;
            assert(pmatrix_index < network.num_branches());
            network.edges[pmatrix_index].init(pmatrix_index, nullptr, nullptr, rnode->length);
            Link *linkToParent = make_link(&network.nodes[clv_index], &network.edges[pmatrix_index],
                    Direction::INCOMING);
            network.edges[pmatrix_index].link1 = linkToParent;
            assert(rnode->children.size() == 2);
        } else {
            assert(rnode->children.size() == 3);
        }
    }

    for (size_t i = 0; i < rnetwork_reticulations.size(); ++i) {
        RootedNetworkNode *rnode = rnetwork_reticulations[i];
        size_t clv_index = i + rnetwork_tips.size() + rnetwork_inner_tree.size();
        rnode->clv_index = clv_index;
        rnode->reticulation_index = i;
        int scaler_index = i + rnetwork_inner_tree.size();
        ReticulationData retData;
        retData.init(i, rnode->reticulationName, false, nullptr, nullptr, nullptr);
        assert(clv_index < network.num_nodes());
        network.nodes[clv_index].initReticulation(clv_index, scaler_index, rnode->label, retData);

        size_t pmatrix_index = rnetwork_tips.size() + rnetwork_inner_tree.size() - 1 + 2 * i;
        assert(pmatrix_index + 1 < network.num_branches());
        network.edges[pmatrix_index].init(pmatrix_index, nullptr, nullptr, rnode->firstParentLength,
                rnode->firstParentProb);
        network.edges[pmatrix_index + 1].init(pmatrix_index + 1, nullptr, nullptr, rnode->secondParentLength,
                rnode->secondParentProb);
        network.reticulation_nodes[i] = &network.nodes[clv_index];

        Link *linkToFirstParent = make_link(&network.nodes[clv_index], &network.edges[pmatrix_index],
                Direction::INCOMING);
        network.edges[pmatrix_index].link1 = linkToFirstParent;

        Link *linkToSecondParent = make_link(&network.nodes[clv_index], &network.edges[pmatrix_index + 1],
                Direction::INCOMING);
        network.edges[pmatrix_index + 1].link1 = linkToSecondParent;
    }

    // 2.) Create all the outgoing links
    for (const auto &rnode : rnetwork_nodes) {
        if (rnode == rnetwork.root) {
            continue;
        }
        if (rnode->isReticulation) { // 2 parents
            size_t pmatrix_index = rnetwork_tips.size() + rnetwork_inner_tree.size() - 1
                    + 2 * rnode->reticulation_index;

            assert(rnode->firstParent->clv_index < network.num_nodes());
            assert(rnode->secondParent->clv_index < network.num_nodes());
            assert(pmatrix_index + 1 < network.num_branches());

            Link *linkFromFirstParent = make_link(&network.nodes[rnode->firstParent->clv_index],
                    &network.edges[pmatrix_index], Direction::OUTGOING);
            Link *linkFromSecondParent = make_link(&network.nodes[rnode->secondParent->clv_index],
                    &network.edges[pmatrix_index + 1], Direction::OUTGOING);

            assert(pmatrix_index + 1 < network.num_branches());
            network.edges[pmatrix_index].link2 = linkFromFirstParent;
            network.edges[pmatrix_index + 1].link2 = linkFromSecondParent;
        } else { // 1 parent
            size_t pmatrix_index = rnode->clv_index;

            assert(rnode->parent->clv_index < network.num_nodes());
            assert(pmatrix_index < network.num_branches());

            Link *linkFromParent = make_link(&network.nodes[rnode->parent->clv_index],
                    &network.edges[pmatrix_index], Direction::OUTGOING);
            network.edges[pmatrix_index].link2 = linkFromParent;
        }
    }

    // 3.) Create the outer links
    for (const auto &rnode : rnetwork_nodes) {
        if (rnode == rnetwork.root) {
            continue;
        }
        if (rnode->isReticulation) { // 2 parents
            size_t pmatrix_index = rnetwork_tips.size() + rnetwork_inner_tree.size() - 1
                    + 2 * rnode->reticulation_index;
            Link *linkFromFirstParent = network.edges[pmatrix_index].link2;
            Link *linkFromSecondParent = network.edges[pmatrix_index + 1].link2;

            Link *linkToFirstParent = getLinkToClvIndex(&network.nodes[rnode->clv_index],
                    rnode->firstParent->clv_index);
            Link *linkToSecondParent = getLinkToClvIndex(&network.nodes[rnode->clv_index],
                    rnode->secondParent->clv_index);

            linkFromFirstParent->outer = linkToFirstParent;
            linkToFirstParent->outer = linkFromFirstParent;
            linkFromSecondParent->outer = linkToSecondParent;
            linkToSecondParent->outer = linkFromSecondParent;

            network.nodes[rnode->clv_index].getReticulationData()->link_to_first_parent = linkToFirstParent;
            network.nodes[rnode->clv_index].getReticulationData()->link_to_second_parent = linkToSecondParent;

            // also set the link to the reticulation child node
            Link *linkToChild = linkToSecondParent->next;
            assert(linkToChild && linkToChild != linkToFirstParent);
            network.nodes[rnode->clv_index].getReticulationData()->link_to_child = linkToChild;
        } else { // 1 parent
            size_t pmatrix_index = rnode->clv_index;
            Link *linkFromParent = network.edges[pmatrix_index].link2;

            Link *linkToParent = getLinkToClvIndex(&network.nodes[rnode->clv_index], rnode->parent->clv_index);
            linkFromParent->outer = linkToParent;
            linkToParent->outer = linkFromParent;
        }
    }

    // TODO: Update changed memory layout in the Google Doc
    network.root = &network.nodes[network.num_nodes() - 1 - rnetwork.reticulationCount];

    return network;
}

std::pair<size_t, size_t> makeToplevelTrifurcation(RootedNetwork &rnetwork) {
    RootedNetworkNode *root = rnetwork.root;
    size_t node_count = rnetwork.nodes.size();
    size_t branch_count = rnetwork.branchCount;
    if (root->children.size() == 3) {
        return std::make_pair(node_count, branch_count);
    } else if (root->children.size() > 3) {
        throw std::runtime_error("The network is not bifurcating");
    }
    // special case: check if rnetwork.root has only one child... if so, reset the root to its child.
    while (root->children.size() == 1) {
        root = root->children[0];
        node_count--;
        branch_count--;
    }

    if (root->children.size() == 2) { // make it trifurcating
        unsigned int newRootChildIdx = 0;
        if (root->children[0]->children.size() == 0) {
            newRootChildIdx = 1;
        }
        RootedNetworkNode *new_root = root->children[newRootChildIdx];
        new_root->children.push_back(root->children[!newRootChildIdx]);
        root->children[!newRootChildIdx]->length += new_root->length;
        root->children[!newRootChildIdx]->parent = new_root;
        node_count--;
        branch_count--;
        root = new_root;
        root->parent = nullptr;
        rnetwork.root = root;
    }

    return std::make_pair(node_count, branch_count);
}

Network convertNetwork(RootedNetwork &rnetwork, int maxReticulations) {
    //std::cout << exportDebugInfo(rnetwork) << "\n";
    std::pair<size_t, size_t> node_and_branch_count = makeToplevelTrifurcation(rnetwork);
    //std::cout << exportDebugInfo(rnetwork) << "\n";
    size_t node_count = node_and_branch_count.first;
    size_t branch_count = node_and_branch_count.second;

    Network network;
    network = convertNetworkToplevelTrifurcation(rnetwork, node_count, branch_count, maxReticulations);
    assert(!network.root->isTip());

    // ensure that no branch lengths are zero
    for (size_t i = 0; i < network.num_branches(); ++i) {
        assert(network.edges[i].length != 0);
    }

    //std::cout << exportDebugInfo(network) << "\n";

    assert(networkIsConnected(network));

    return network;
}

Network readNetworkFromString(const std::string &newick, int maxReticulations) {
    RootedNetwork *rnetwork = parseRootedNetworkFromNewickString(newick);
    Network network = convertNetwork(*rnetwork, maxReticulations);
    delete rnetwork;
    return network;
}

Network readNetworkFromFile(const std::string &filename, int maxReticulations) {
    std::ifstream t(filename);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string newick = buffer.str();
    return readNetworkFromString(newick, maxReticulations);
}

std::string newickNodeName(const Node *node, const Node *parent) {
    assert(node);
    std::stringstream sb("");

    sb << node->label;
    if (node->getType() == NodeType::RETICULATION_NODE) {
        assert(parent);
        sb << "#" << node->getReticulationData()->getLabel();
        Link *link = node->getReticulationData()->getLinkToFirstParent();
        double prob = getReticulationFirstParentProb(node);
        if (getReticulationSecondParent(node) == parent) {
            link = node->getReticulationData()->getLinkToSecondParent();
            prob = 1.0 - prob;
        } else {
            assert(getReticulationFirstParent(node) == parent);
        }

        sb << ":" << link->edge->length << ":";
        if (link->edge->support != 0.0) {
            sb << link->edge->support;
        }
        sb << ":" << prob;
    } else {
        if (parent != nullptr) {
            sb << ":" << getEdgeTo(node, parent)->length;
            if (getEdgeTo(node, parent)->support != 0.0) {
                sb << ":" << getEdgeTo(node, parent)->support;
            }
        }
    }
    return sb.str();
}

std::string printNodeNewick(Node *node, Node *parent, std::unordered_set<Node*> &visited_reticulations) {
    std::stringstream sb("");
    std::vector<Node*> children = getChildren(node, parent);
    if (!children.empty() && visited_reticulations.find(node) == visited_reticulations.end()) {
        sb << "(";
        for (size_t i = 0; i < children.size() - 1; i++) {
            sb << printNodeNewick(children[i], node, visited_reticulations);
            sb << ",";
        }
        sb << printNodeNewick(children[children.size() - 1], node, visited_reticulations);
        sb << ")";
        if (node->getType() == NodeType::RETICULATION_NODE) {
            visited_reticulations.insert(node);
        }
    }
    sb << newickNodeName(node, parent);
    return sb.str();
}

std::string toExtendedNewick(Network &network) {
    std::unordered_set<Node*> visited_reticulations;
    return printNodeNewick(network.root, nullptr, visited_reticulations) + ";";
}

}
