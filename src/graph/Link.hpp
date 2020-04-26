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

struct Node;
struct Edge;

struct Link { // subnode in raxml-ng
    void init(Node *node, Edge *edge, Link *next, Link *outer, Direction direction) {
        this->node = node;
        this->edge = edge;
        this->next = next;
        this->outer = outer;
        this->direction = direction;
    }

    Node *node = nullptr;
    Edge *edge = nullptr;

    Link *next = nullptr;
    Link *outer = nullptr;
    Direction direction = Direction::INCOMING;
};
}
