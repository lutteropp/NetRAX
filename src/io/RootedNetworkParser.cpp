/*
 * SimpleNewickParser.hpp
 *
 *  Created on: Sep 26, 2019
 *      Author: Sarah Lutteropp
 */

// Adapted from https://stackoverflow.com/a/41418573
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <cassert>

#include "RootedNetworkParser.hpp"

namespace netrax {

// returns a substring from beginIndex to endIndex - 1
inline std::string substring(const std::string& str, size_t beginIndex, size_t endIndex) {
	size_t len = endIndex - beginIndex;
	return str.substr(beginIndex, len);
}

std::string parseReticulationNameFromString(const std::string& str) {
	size_t hashtagIndex = str.find('#');
	if (hashtagIndex == std::string::npos) {
		return "";
	} else {
		size_t firstColonIndex = str.find(':');
		assert(firstColonIndex == std::string::npos || firstColonIndex > hashtagIndex);
		std::string res = substring(str, hashtagIndex + 1, firstColonIndex);
		if (res == "") {
			throw std::runtime_error("Node is a reticulation, but no reticulation name is given!");
		} else {
			return res;
		}
	}
}

inline double tolerantSTOD(const std::string& str) {
	if (str.empty()) {
		return 0.0;
	} else {
		return std::stod(str);
	}
}

std::array<double, 3> readBrlenSupportProb(const std::string& str) {
	std::array<double, 3> res = { 0, 0, 0 };
	size_t colonCount = std::count(str.begin(), str.end(), ':');
	assert(colonCount <= 3);
	if (colonCount == 0) {
		return res;
	}

	std::array<size_t, 3> colonPos = { std::string::npos, std::string::npos, std::string::npos };
	colonPos[0] = str.find(':');
	for (size_t i = 1; i < colonCount; ++i) {
		colonPos[i] = str.find(':', colonPos[i - 1] + 1);
	}
	if (colonCount > 0) {
		res[0] = tolerantSTOD(substring(str, colonPos[0] + 1, colonPos[1]));
	}
	if (colonCount > 1) {
		res[1] = tolerantSTOD(substring(str, colonPos[1] + 1, colonPos[2]));
	}
	if (colonCount > 2) {
		res[2] = tolerantSTOD(str.substr(colonPos[2] + 1));
	}
	return res;
}

RootedNetworkNode* buildNormalNodeFromString(const std::string& str, RootedNetworkNode* parent) {
	RootedNetworkNode* node = new RootedNetworkNode();
	node->isReticulation = false;
	node->parent = parent;

	size_t firstColonIndex = str.find(':');
	node->label = substring(str, 0, firstColonIndex);
	std::array<double, 3> brlen_support_prob = readBrlenSupportProb(str);
	node->length = brlen_support_prob[0];
	node->support = brlen_support_prob[1];

	return node;
}

RootedNetworkNode* buildNewReticulationNodeFromString(const std::string& str, size_t reticulationId, RootedNetworkNode* firstParent) {
	// case 1: X#H1:brlen:support:prob
	// case 2: X#H1:brlen:support
	// case 3: X#H1:brlen
	// case 4: X#H1
	// X can also be empty, as well as brlen, support and prob (if no the last ones).
	RootedNetworkNode* node = new RootedNetworkNode();
	node->isReticulation = true;
	assert(firstParent);
	size_t hashtagIndex = str.find('#');
	assert(hashtagIndex != std::string::npos);
	size_t colonCount = std::count(str.begin(), str.end(), ':');
	assert(colonCount <= 3);

	node->label = substring(str, 0, hashtagIndex);
	size_t firstColonPos = str.find(':');
	assert(firstColonPos > hashtagIndex);
	std::string reticulationName = substring(str, hashtagIndex + 1, firstColonPos);
	node->reticulationName = reticulationName;
	assert(!node->reticulationName.empty());
	node->reticulationId = reticulationId;
	node->firstParent = firstParent;

	std::array<double, 3> brlen_support_prob = readBrlenSupportProb(str);
	node->firstParentLength = brlen_support_prob[0];
	node->firstParentSupport = brlen_support_prob[1];
	node->firstParentProb = brlen_support_prob[2];
	return node;
}

void extendReticulationNodeFromString(const std::string& str, RootedNetworkNode* node, RootedNetworkNode* secondParent) {
	// case 1: X#H1:brlen:support:prob
	// case 2: X#H1:brlen:support
	// case 3: X#H1:brlen
	// case 4: X#H1
	// X can also be empty, as well as brlen, support and prob (if no the last ones).
	assert(node);
	assert(node->isReticulation);
	size_t hashtagIndex = str.find('#');
	assert(hashtagIndex != std::string::npos);
	size_t colonCount = std::count(str.begin(), str.end(), ':');
	assert(colonCount <= 3);

	std::string name = substring(str, 0, hashtagIndex);
	size_t firstColonPos = str.find(':');
	assert(firstColonPos > hashtagIndex);
	std::string reticulationName = substring(str, hashtagIndex + 1, firstColonPos);
	assert(!reticulationName.empty());
	assert(node->label == name);
	assert(node->reticulationName == reticulationName);

	std::array<double, 3> brlen_support_prob = readBrlenSupportProb(str);
	node->secondParentLength = brlen_support_prob[0];
	node->secondParentSupport = brlen_support_prob[1];
	node->secondParentProb = brlen_support_prob[2];
	node->secondParent = secondParent;
}

RootedNetworkNode* buildNodeFromString(const std::string& str, RootedNetworkNode* parent, std::vector<RootedNetworkNode*>& nodeList,
		std::unordered_map<std::string, size_t>& reticulations, size_t* num_reticulations) {
	size_t hashtagIndex = str.find('#');
	if (hashtagIndex != std::string::npos) {
		std::string reticulationName = parseReticulationNameFromString(str);
		if (reticulations.find(reticulationName) != reticulations.end()) {
			extendReticulationNodeFromString(str, nodeList[reticulations[reticulationName]], parent);
			return nodeList[reticulations[reticulationName]];
		} else {
			RootedNetworkNode* retNode = buildNewReticulationNodeFromString(str, *num_reticulations, parent);
			(*num_reticulations)++;
			nodeList.push_back(retNode);
			reticulations[reticulationName] = nodeList.size() - 1;
			return nodeList[nodeList.size() - 1];
		}
	} else {
		RootedNetworkNode* node = buildNormalNodeFromString(str, parent);
		nodeList.push_back(node);
		return nodeList[nodeList.size() - 1];
	}
}

std::vector<std::string> split(const std::string& s) {
	std::vector<size_t> splitIndices;
	size_t rightParenCount = 0;
	size_t leftParenCount = 0;
	for (size_t i = 0; i < s.length(); i++) {
		switch (s[i]) {
		case '(':
			leftParenCount++;
			break;
		case ')':
			rightParenCount++;
			break;
		case ',':
			if (leftParenCount == rightParenCount)
				splitIndices.push_back(i);
			break;
		}
	}
	size_t numSplits = splitIndices.size() + 1;
	std::vector<std::string> splits(numSplits);
	if (numSplits == 1) {
		splits[0] = s;
	} else {
		splits[0] = substring(s, 0, splitIndices.at(0));
		for (size_t i = 1; i < splitIndices.size(); i++) {
			splits[i] = substring(s, splitIndices.at(i - 1) + 1, splitIndices.at(i));
		}
		splits[numSplits - 1] = s.substr(splitIndices.at(splitIndices.size() - 1) + 1);
	}
	return splits;
}

RootedNetworkNode* readSubtree(RootedNetworkNode* parent, const std::string& s, std::vector<RootedNetworkNode*>& nodeList,
		std::unordered_map<std::string, size_t>& reticulations_lookup, size_t* num_reticulations, size_t* num_tips) {
	size_t leftParen = s.find('(');
	size_t rightParen = s.rfind(')');
	if (leftParen != std::string::npos && rightParen != std::string::npos) {
		std::string name = s.substr(rightParen + 1);
		std::vector<std::string> childrenString = split(substring(s, leftParen + 1, rightParen));

		RootedNetworkNode* node = buildNodeFromString(name, parent, nodeList, reticulations_lookup, num_reticulations);

		for (std::string sub : childrenString) {
			RootedNetworkNode* child = readSubtree(node, sub, nodeList, reticulations_lookup, num_reticulations, num_tips);
			node->children.push_back(child);
			if (!child->isReticulation) {
				child->parent = node;
			}
		}
		return node;
	} else if (leftParen == rightParen) {
		RootedNetworkNode* node = buildNodeFromString(s, parent, nodeList, reticulations_lookup, num_reticulations);
		if (!node->isReticulation) {
			node->tip_index = *num_tips;
			(*num_tips)++;
		}
		return node;
	} else
		throw std::runtime_error("unbalanced ()'s");
}

RootedNetwork parseRootedNetworkFromNewickString(const std::string& newick) {
	RootedNetwork rnetwork;
	// TODO: special case: ignore faulty extra Céline parantheses which lead to top-level monofurcation
	std::unordered_map<std::string, size_t> reticulations_lookup;

	size_t semicolonPos = newick.find(';');
	assert(semicolonPos != std::string::npos);
	assert(std::count(newick.begin(), newick.end(), ';') == 1);

	rnetwork.root = readSubtree(nullptr, substring(newick, 0, semicolonPos), rnetwork.nodes, reticulations_lookup,
			&rnetwork.reticulationCount, &rnetwork.tipCount);
	return rnetwork;
}

std::string newickNodeName(const RootedNetworkNode* node, const RootedNetworkNode* parent) {
	assert(node);
	std::stringstream sb("");

	sb << node->label;
	if (node->isReticulation) {
		assert(parent);
		sb << "#" << node->reticulationName;
		if (node->firstParent == parent) {
			sb << ":" << node->firstParentLength << ":" << node->firstParentSupport << ":" << node->firstParentProb;
		} else {
			assert(node->secondParent == parent);
			sb << ":" << node->secondParentLength << ":" << node->secondParentSupport << ":" << node->secondParentProb;
		}
	} else {
		sb << ":" << node->length << ":" << node->support;
	}
	return sb.str();
}

std::string printNodeNewick(const RootedNetworkNode* node, const RootedNetworkNode* parent, std::unordered_set<const RootedNetworkNode*>& visited_reticulations) {
	std::stringstream sb("");
	if (!node->children.empty() && visited_reticulations.find(node) == visited_reticulations.end()) {
		sb << "(";
		for (size_t i = 0; i < node->children.size() - 1; i++) {
			sb << printNodeNewick(node->children[i], node, visited_reticulations);
			sb << ",";
		}
		sb << printNodeNewick(node->children[node->children.size() - 1], node, visited_reticulations);
		sb << ")";
		if (node->isReticulation) {
			visited_reticulations.insert(node);
		}
	}
	sb << newickNodeName(node, parent);
	return sb.str();
}

std::string toNewickString(const RootedNetwork& network) {
	std::unordered_set<const RootedNetworkNode*> visited_reticulations;
	return printNodeNewick(network.root, nullptr, visited_reticulations) + ";";
}

}

