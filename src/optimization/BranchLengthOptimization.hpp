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
double optimize_branches(AnnotatedNetwork &ann_network, double min_brlen, double max_brlen, double lh_epsilon,
        int max_iters, int opt_method, int radius);

double optimize_branches(AnnotatedNetwork &ann_network, double min_brlen, double max_brlen, double lh_epsilon,
        int max_iters, std::unordered_set<size_t> &candidates);

double optimize_branches(AnnotatedNetwork &ann_network, double min_brlen, double max_brlen, double lh_epsilon,
        int max_iters);
}
