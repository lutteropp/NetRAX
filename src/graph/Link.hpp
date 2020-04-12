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
#include "Edge.hpp"

namespace netrax {

struct Node;
struct Link { // subnode in raxml-ng
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
		if (edge->link1 == this) {
			return edge->link2->node;
		} else {
			assert(edge->link2 == this);
			return edge->link1->node;
		}
	}

	size_t node_index = 0;
	Node* node = nullptr;
	Edge* edge = nullptr;

	Link* next = nullptr;
	Link* outer = nullptr;
	Direction direction = Direction::INCOMING;
};
}
