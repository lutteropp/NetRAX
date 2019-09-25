/*
 * Network.hpp
 *
 *  Created on: Sep 2, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace netrax {

class Node;
class Edge;
class Link;

enum class Direction {
	UNDEFINED, INCOMING, OUTGOING
};

enum class NodeType {
	BASIC_NODE, RETICULATION_NODE
};

class ReticulationData {
public:
	ReticulationData() :
			reticulation_index(0), label(""), active_parent(0), link_to_first_parent(nullptr), link_to_second_parent(nullptr), link_to_child(
					nullptr), prob(0.5) {
	}

	void init(size_t index, const std::string& label, bool activeParent, Link* linkToFirstParent, Link* linkToSecondParent,
			Link* linkToChild, double prob) {
		this->reticulation_index = index;
		this->label = label;
		this->active_parent = activeParent;
		this->link_to_first_parent = linkToFirstParent;
		this->link_to_second_parent = linkToSecondParent;
		this->link_to_child = linkToChild;
		this->prob = prob;
	}

	ReticulationData(const ReticulationData& retData) {
		reticulation_index = retData.reticulation_index;
		label = retData.label;
		active_parent = retData.active_parent;
		link_to_first_parent = retData.link_to_first_parent;
		link_to_second_parent = retData.link_to_second_parent;
		link_to_child = retData.link_to_child;
		prob = retData.prob;
	}

	size_t getReticulationIndex() const {
		return reticulation_index;
	}
	const std::string& getLabel() const {
		return label;
	}
	Link* getLinkToActiveParent() const {
		if (active_parent == 0) {
			return link_to_first_parent;
		} else {
			return link_to_second_parent;
		}
	}
	void setActiveParent(bool val) {
		active_parent = val;
	}
	double getProb() const {
		return prob;
	}
	double getActiveProb() const {
		if (active_parent == 0) {
			return prob;
		} else {
			return 1.0 - prob;
		}
	}
	void setProb(double val) {
		prob = val;
	}
	Link* getLinkToFirstParent() const {
		return link_to_first_parent;
	}
	Link* getLinkToSecondParent() const {
		return link_to_second_parent;
	}
	Link* getLinkToChild() const {
		return link_to_child;
	}
	void setLinkToFirstParent(Link* link) {
		link_to_first_parent = link;
	}
	void setLinkToSecondParent(Link* link) {
		link_to_second_parent = link;
	}
	void setLinkToChild(Link* link) {
		link_to_child = link;
	}
private:
	size_t reticulation_index;
	std::string label;
	bool active_parent; // 0: first_parent, 1: second_parent
	Link* link_to_first_parent; // The link that has link->outer->node as the first parent
	Link* link_to_second_parent; // The link that has link->outer->node as the second parent
	Link* link_to_child; // The link that has link->outer->node as the child
	double prob; // probability of taking the first parent
};

struct Link { // subnode in raxml-ng
	Link() :
			node_index(0), node(nullptr), edge(nullptr), next(nullptr), outer(nullptr), direction(Direction::UNDEFINED) {
	}

	void init(size_t index, Node* node, Edge* edge, Link* next, Link* outer, Direction direction) {
		this->node_index = index;
		this->node = node;
		this->edge = edge;
		this->next = next;
		this->outer = outer;
		this->direction = direction;
	}

	Node* getTargetNode() const {
		assert(outer != nullptr);
		return outer->node;
	}

	size_t node_index;
	Node* node;
	Edge* edge;

	Link* next;
	Link* outer;
	Direction direction;
};

class Node {
public:
	Node() :
			clv_index(0), scaler_index(-1), link(nullptr), type(NodeType::BASIC_NODE), reticulationData(nullptr) {

	}

	void initBasic(size_t index, int scaler_index, Link* link, const std::string& label) {
		this->clv_index = index;
		this->scaler_index = scaler_index;
		this->link = link;
		this->label = label;
		this->type = NodeType::BASIC_NODE;
		this->reticulationData = nullptr;
	}
	void initReticulation(size_t index, int scaler_index, Link* link, const std::string& label, const ReticulationData& retData) {
		this->clv_index = index;
		this->scaler_index = scaler_index;
		this->link = link;
		this->label = label;
		this->type = NodeType::RETICULATION_NODE;
		reticulationData = std::make_unique<ReticulationData>(retData);
	}

	bool isTip() {
		assert(this->link);
		return (!this->link->next);
	}

	std::vector<Node*> getChildren(Node* myParent) const {
		std::vector<Node*> children;
		if (type == NodeType::RETICULATION_NODE) {
			children.push_back(reticulationData->getLinkToChild()->getTargetNode());
		} else { // normal node
			std::vector<Node*> neighbors = getNeighbors();
			for (size_t i = 0; i < neighbors.size(); ++i) {
				if (neighbors[i] != myParent) {
					children.push_back(neighbors[i]);
				}
			}
		}
		return children;
	}

	std::vector<Node*> getActiveChildren(Node* myParent) const {
		std::vector<Node*> activeChildren;
		std::vector<Node*> children = getChildren(myParent);
		for (size_t i = 0; i < children.size(); ++i) {
			if (children[i]->getType() == NodeType::RETICULATION_NODE) {
				// we need to check if the child is active, this is, if we are currently the selected parent
				if (children[i]->getReticulationData()->getLinkToActiveParent()->getTargetNode() != this) {
					continue;
				}
			}
			activeChildren.push_back(children[i]);
		}
		assert(activeChildren.size() <= 2 || (myParent == nullptr && activeChildren.size() == 3));
		return activeChildren;
	}

	std::vector<Node*> getNeighbors() const {
		std::vector<Node*> neighbors;
		Link* currLink = link;
		do {
			Node* target = currLink->getTargetNode();
			if (std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
				throw std::runtime_error("Loop in neighbors list!");
			}
			neighbors.push_back(target);
			if (!currLink->next) { // leaf node
				assert(neighbors.size() == 1);
				break;
			}
			currLink = currLink->next;
		} while (currLink != link);
		return neighbors;
	}

	Edge* getEdgeTo(Node* target) const {
		assert(target);
		Link* currLink = link;
		while (currLink != nullptr) {
			if (currLink->outer->node == target) {
				return currLink->edge;
			}
			currLink = currLink->next;
			if (currLink == link) {
				break;
			}
		}
		throw std::runtime_error("The given target node is not a neighbor of this node");
	}

	size_t getClvIndex() const {
		return clv_index;
	}

	void setClvIndex(size_t index) {
		this->clv_index = index;
	}

	int getScalerIndex() const {
		return scaler_index;
	}

	void setScalerIndex(int idx) {
		scaler_index = idx;
	}

	Link* getLink() const {
		return link;
	}
	void setLink(Link* link) {
		this->link = link;
	}
	const std::string& getLabel() const {
		return label;
	}
	void setLabel(const std::string& label) {
		this->label = label;
	}
	NodeType getType() const {
		return type;
	}

	const std::unique_ptr<ReticulationData>& getReticulationData() const {
		assert(type == NodeType::RETICULATION_NODE);
		return reticulationData;
	}
private:
	size_t clv_index;
	int scaler_index;
	Link* link;
	std::string label;
	NodeType type;

	std::unique_ptr<ReticulationData> reticulationData;
};

class Edge {
public:
	Edge() :
			pmatrix_index(0), link1(nullptr), link2(nullptr), length(0.0) {
	}

	void init(size_t index, Link* link1, Link* link2, double length) {
		this->pmatrix_index = index;
		this->link1 = link1;
		this->link2 = link2;
		this->length = length;
	}

	size_t getPMatrixIndex() const {
		return pmatrix_index;
	}
	Link* getLink1() const {
		return link1;
	}
	Link* getLink2() const {
		return link2;
	}
	double getLength() const {
		return length;
	}
	void setLink1(Link* link) {
		link1 = link;
	}
	void setLink2(Link* link) {
		link2 = link;
	}
	void setLength(double length) {
		this->length = length;
	}
private:
	size_t pmatrix_index;
	Link* link1;
	Link* link2;

	double length;
};

class Network {
public:
	Network() :
			root(nullptr) {
	}

	std::vector<double> collectBranchLengths() const {
		std::vector<double> brLengths(edges.size());
		for (size_t i = 0; i < edges.size(); ++i) {
			brLengths[i] = edges[i].getLength();
		}
		return brLengths;
	}
	void applyBranchLengths(const std::vector<double>& branchLengths) {
		assert(branchLengths.size() == edges.size());
		for (size_t i = 0; i < edges.size(); ++i) {
			edges[i].setLength(branchLengths[i]);
		}
	}
	void setReticulationParents(size_t treeIdx) {
		for (size_t i = 0; i < reticulation_nodes.size(); ++i) {
			// check if i-th bit is set in treeIdx
			bool activeParentIdx = treeIdx & (1 << i);
			reticulation_nodes[i]->getReticulationData()->setActiveParent(activeParentIdx);
		}
	}

	Node* getNodeByLabel(const std::string& label) {
		Node* result = nullptr;
		for (size_t i = 0; i < nodes.size(); ++i) {
			if (nodes[i].getLabel() == label) {
				result = &nodes[i];
				break;
			}
		}
		assert(result);
		return result;
	}

	size_t num_tips() const {
		return tip_nodes.size();
	}

	size_t num_inner() const {
		return nodes.size() - tip_nodes.size();
	}

	size_t num_branches() const {
		return edges.size();
	}

	size_t num_reticulations() const {
		return reticulation_nodes.size();
	}

	std::vector<Node> nodes;
	std::vector<Edge> edges;
	std::vector<Link> links;
	std::vector<Node*> reticulation_nodes;
	std::vector<Node*> tip_nodes;
	std::vector<Node*> inner_nodes;
	Node* root;
};

}
