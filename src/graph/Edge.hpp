/*
 * Edge.hpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <cstddef>
#include <limits>

namespace netrax {

struct Link;
struct Node;
struct Edge {
  void init(size_t index, Link *link1, Link *link2, double length,
            double prob = 1.0) {
    this->pmatrix_index = index;
    this->link1 = link1;
    this->link2 = link2;
    this->length = length;
    this->prob = prob;
  }

  void clear() {
    pmatrix_index = std::numeric_limits<size_t>::max();
    link1 = nullptr;
    link2 = nullptr;
    length = 0.0;
    support = 0.0;
    prob = 1.0;
  }

  size_t pmatrix_index = std::numeric_limits<size_t>::max();
  Link *link1 = nullptr;
  Link *link2 = nullptr;
  double length = 0.0;
  double support = 0.0;
  double prob = 1.0;
};
}  // namespace netrax
