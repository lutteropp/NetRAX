/*
 * Traversal.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <vector>
extern "C" {
#include <libpll/pll.h>
}
#include "../graph/Network.hpp"

namespace netrax {

std::vector<Node*> postorderTraversal(const Network &network, size_t tree_index);

}
