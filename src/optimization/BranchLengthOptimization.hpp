/*
 * BranchLengthOptimization.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../graph/Network.hpp"
#include "../NetraxOptions.hpp"

extern "C" {
#include <libpll/pll_tree.h>
}

namespace netrax {
double optimize_branches(const NetraxOptions& options, Network& network, pllmod_treeinfo_t& fake_treeinfo, double min_brlen, double max_brlen, double lh_epsilon,
		int max_iters, int opt_method, int radius);
}
