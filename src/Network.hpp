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
			index(0), label(""), active_parent(0), link_to_first_parent(nullptr), link_to_second_parent(nullptr), link_to_child(nullptr), prob(
					0.5) {
	}

	void init(size_t index, const std::string& label, bool activeParent, Link* linkToFirstParent, Link* linkToSecondParent,
			Link* linkToChild, double prob) {
		this->index = index;
		this->label = label;
		this->active_parent = activeParent;
		this->link_to_first_parent = linkToFirstParent;
		this->link_to_second_parent = linkToSecondParent;
		this->link_to_child = linkToChild;
		this->prob = prob;
	}

	ReticulationData(const ReticulationData& retData) {
		index = retData.index;
		label = retData.label;
		active_parent = retData.active_parent;
		link_to_first_parent = retData.link_to_first_parent;
		link_to_second_parent = retData.link_to_second_parent;
		link_to_child = retData.link_to_child;
		prob = retData.prob;
	}

	size_t getIndex() const {
		return index;
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
	size_t index;
	std::string label;
	bool active_parent; // 0: first_parent, 1: second_parent
	Link* link_to_first_parent; // The link that has link->outer->node as the first parent
	Link* link_to_second_parent; // The link that has link->outer->node as the second parent
	Link* link_to_child; // The link that has link->outer->node as the child
	double prob; // probability of taking the first parent
};

struct Link { // subnode in raxml-ng
	Link() :
			index(0), node(nullptr), edge(nullptr), next(nullptr), outer(nullptr), direction(Direction::UNDEFINED) {
	}

	void init(size_t index, Node* node, Edge* edge, Link* next, Link* outer, Direction direction) {
		this->index = index;
		this->node = node;
		this->edge = edge;
		this->next = next;
		this->outer = outer;
		this->direction = direction;
	}

	size_t index; // node_index in libpll
	Node* node;
	Edge* edge;

	Link* next;
	Link* outer;
	Direction direction;
};

class Node {
public:
	Node() :
			index(0), scaler_index(-1), link(nullptr), type(NodeType::BASIC_NODE), reticulationData(nullptr) {

	}

	void initBasic(size_t index, int scaler_index, Link* link, const std::string& label) {
		this->index = index;
		this->scaler_index = scaler_index;
		this->link = link;
		this->label = label;
		this->type = NodeType::BASIC_NODE;
		this->reticulationData = nullptr;
	}
	void initReticulation(size_t index, int scaler_index, Link* link, const std::string& label, const ReticulationData& retData) {
		this->index = index;
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

	std::vector<Node*> getNeighbors() const {
		std::vector<Node*> neighbors;
		Link* currLink = link;
		while (currLink != nullptr) {
			neighbors.push_back(currLink->node);
			currLink = currLink->next;
			if (currLink == link) {
				break;
			}
		}
		return neighbors;
	}

	size_t getIndex() const {
		return index;
	}

	void setIndex(size_t index) {
		this->index = index;
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
	size_t index; // clv_index in libpll
	int scaler_index;
	Link* link;
	std::string label;
	NodeType type;

	std::unique_ptr<ReticulationData> reticulationData;
};

class Edge {
public:
	Edge() :
			index(0), link1(nullptr), link2(nullptr), length(0.0) {
	}

	void init(size_t index, Link* link1, Link* link2, double length) {
		this->index = index;
		this->link1 = link1;
		this->link2 = link2;
		this->length = length;
	}

	size_t getIndex() const {
		return index;
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
	size_t index; // pmatrix_index in libpll
	Link* link1;
	Link* link2;

	double length;
};

class Network {
public:
	Network() :
			root(nullptr), tip_count(0) {
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

	std::vector<Node> nodes;
	std::vector<Edge> edges;
	std::vector<Link> links;
	std::vector<Node*> reticulation_nodes;
	Node* root;
	size_t tip_count;
};

}
