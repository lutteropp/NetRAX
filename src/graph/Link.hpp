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

struct Link { // subnode in raxml-ng
    void init(unsigned int node_clv_index, unsigned int edge_pmatrix_index, Link *next, Link *outer,
            Direction direction) {
        this->link_clv_index = node_clv_index;
        this->edge_pmatrix_index = edge_pmatrix_index;
        this->next = next;
        this->outer = outer;
        this->direction = direction;
    }

    unsigned int link_clv_index = 0;
    unsigned int edge_pmatrix_index = 0;

    Link *next = nullptr;
    Link *outer = nullptr;
    Direction direction = Direction::INCOMING;
};
}
