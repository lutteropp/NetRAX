/*
 * Link.hpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <cassert>
#include <cstddef>

#include "Direction.hpp"

namespace netrax {

class Node;
class Edge;
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

	size_t getNodeIndex() const {
		return node_index;
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
}
