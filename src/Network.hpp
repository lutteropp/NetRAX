/*
 * Network.hpp
 *
 *  Created on: Sep 2, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <string>
#include <vector>
#include <cassert>

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

	std::vector<Node> nodes;
	std::vector<Edge> edges;
	std::vector<Link> links;
	std::vector<Node*> reticulation_nodes;
	Node* root;
};

class ReticulationData {
public:
	ReticulationData(size_t index, const std::string& label) :
			index(index), label(label), active_parent(0), link_to_child(nullptr), link_to_first_parent(nullptr), link_to_second_parent(
					nullptr), prob(0.5) {
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
	void setProb(double val) {
		prob = val;
	}
	Link* getLinkToFirstParent() const {
		return link_to_first_parent;
	}
	Node* getLinkToSecondParent() const {
		return link_to_second_parent;
	}
	Node* getLinkToChild() const {
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
	Link* link_to_child; // The link that has link->outer->node as the child
	Link* link_to_first_parent; // The link that has link->outer->node as the first parent
	Link* link_to_second_parent; // The link that has link->outer->node as the second parent
	double prob; // probability of taking the first parent
};

class Node {
public:
	Node(size_t index, const std::string& label) :
			index(index), link(nullptr), label(label), type(NodeType::BASIC_NODE), reticulationData(nullptr) {
	}
	Node(size_t index, const std::string& label, const ReticulationData& retData) :
			index(index), link(nullptr), label(label), type(NodeType::RETICULATION_NODE), reticulationData(retData) {
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
	Link* getLink() const {
		return link;
	}
	const std::string& getLabel() const {
		return label;
	}
	NodeType getType() const {
		return type;
	}

	const ReticulationData& getReticulationData() const {
		assert(type == NodeType::RETICULATION_NODE);
		return reticulationData;
	}
private:
	size_t index; // clv_index in libpll
	Link* link;
	std::string label;
	NodeType type;

	ReticulationData* reticulationData;
};

class Edge {
public:
	Edge(size_t index, Link* link1, Link* link2, double length, bool directed = false) :
			index(index), link1(link1), link2(link2), length(length) {
		link1->edge = this;
		link2->edge = this;
		link1->outer = link2;
		link2->outer = link1;
		if (directed) {
			link1->direction = Direction::OUTGOING;
			link2->direction = Direction::INCOMING;
		} else {
			link1->direction = Direction::UNDEFINED;
			link2->direction = Direction::UNDEFINED;
		}
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

struct Link { // subnode in raxml-ng
	Link(size_t index, Node* node) :
			index(index), node(node), edge(nullptr), next(nullptr), outer(nullptr), direction(Direction::UNDEFINED) {
	}
	size_t index; // node_index in libpll
	Node* node;
	Edge* edge;

	Link* next;
	Link* outer;
	Direction direction;
};

}
