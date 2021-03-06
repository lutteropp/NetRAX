/*
 * SimpleNewickParser.hpp
 *
 *  Created on: Sep 26, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

// Adapted from https://stackoverflow.com/a/41418573

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <memory>

namespace netrax {

struct RootedNetworkNode {
    // only relevant for non-reticulation nodes
    double length = 0.0; // length to parent
    double support = 0.0; // support of the branch to parent
    RootedNetworkNode *parent = nullptr;

    size_t tip_index = std::numeric_limits<size_t>::max(); // this will be used to later on set the clv_index and the pmatrix_index values of tip nodes
    size_t inner_index = std::numeric_limits<size_t>::max(); // this will be used for the scaler index, the clv index of inner nodes, and the pmatrix indices as well

    size_t clv_index = std::numeric_limits<size_t>::max(); // will be set from outside

    // only relevant for reticulation nodes
    bool isReticulation = false;
    int reticulation_index = -1;
    double firstParentProb = 0.5;
    double secondParentProb = 0.5;
    double firstParentLength = 0.0;
    double secondParentLength = 0.0;
    double firstParentSupport = 0.0;
    double secondParentSupport = 0.0;
    RootedNetworkNode *firstParent = nullptr;
    RootedNetworkNode *secondParent = nullptr;
    std::string reticulationName = "";

    std::string label = "";
    std::vector<RootedNetworkNode*> children;
};

struct RootedNetwork {
    RootedNetworkNode *root;
    size_t reticulationCount = 0;
    size_t tipCount = 0;
    size_t branchCount = 0;
    size_t innerCount = 0;
    std::vector<std::unique_ptr<RootedNetworkNode> > nodes;
};

std::string toNewickString(const RootedNetwork &network);
RootedNetwork* parseRootedNetworkFromNewickString(const std::string &newick);
std::string exportDebugInfoRootedNetwork(const RootedNetwork &network);

}
