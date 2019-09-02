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

namespace netrax {

class Node;
class Edge;
class Link;

enum class Direction {
	UNDEFINED, INCOMING, OUTGOING
};

struct Network {
	std::vector<Node> nodes;
	std::vector<Edge> edges;
	std::vector<Link> links;

	Node* root;
};

struct Node {
	size_t index; // clv_index in libpll
	Link* link;

	std::string label;
};

struct Edge {
	size_t index; // pmatrix_index in libpll
	Link* link1;
	Link* link2;

	double length;
	double prob;
};

struct Link { // subnode in raxml-ng
	size_t index; // node_index in libpll
	Link* next;
	Link* outer;
	Node* node;
	Edge* edge;
	Direction direction;
};

}
