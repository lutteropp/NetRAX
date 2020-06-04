/*
 * BranchLengthOptimization.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../graph/Network.hpp"
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

#include <unordered_set>

extern "C" {
#include <libpll/pll_tree.h>
}

namespace netrax {

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius);
double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius,
        std::unordered_set<size_t> &candidates);

}
