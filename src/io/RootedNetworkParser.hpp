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

#include "../Network.hpp"

namespace netrax {

struct RootedNetworkNode {
	std::string name = "";
	std::vector<RootedNetworkNode*> children;

	// only relevant for non-reticulation nodes
	double length = 0.0;
	double support = 0.0;
	RootedNetworkNode* parent = nullptr;

	// only relevant for reticulation nodes
	bool isReticulation = false;
	std::string reticulationName = "";
	int reticulationId = -1;
	double firstParentProb = 0.5;
	double secondParentProb = 0.5;
	double firstParentLength = 0.0;
	double secondParentLength = 0.0;
	double firstParentSupport = 0.0;
	double secondParentSupport = 0.0;
	RootedNetworkNode* firstParent = nullptr;
	RootedNetworkNode* secondParent = nullptr;
};

struct RootedNetwork {
	RootedNetworkNode* root;
	std::vector<RootedNetworkNode*> nodes;
};

std::string toNewickString(const RootedNetwork& network);
RootedNetwork parseRootedNetworkFromNewickString(const std::string& newick);

}
