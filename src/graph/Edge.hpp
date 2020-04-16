/*
 * Edge.hpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <cstddef>

namespace netrax {

struct Link;
struct Node;
struct Edge {
	void init(size_t index, Link* link1, Link* link2, double length) {
		this->pmatrix_index = index;
		this->link1 = link1;
		this->link2 = link2;
		this->length = length;
	}

	Node* getSource() const {
		if (link1->direction == Direction::OUTGOING) {
			return link1->node;
		} else {
			return link2->node;
		}
	}

	Node* getTarget() const {
		if (link1->direction == Direction::INCOMING) {
			return link1->node;
		} else {
			return link2->node;
		}
	}

	size_t pmatrix_index = 0;
	Link* link1 = nullptr;
	Link* link2 = nullptr;
	double length = 0.0;
	double support = 0.0;
};
}
