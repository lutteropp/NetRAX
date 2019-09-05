/*
 * Traversal.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <vector>
#include <libpll/pll.h>

#include "../Network.hpp"

namespace netrax {

std::vector<Node*> postorderTraversal(const Network& network, size_t tree_index);

}
