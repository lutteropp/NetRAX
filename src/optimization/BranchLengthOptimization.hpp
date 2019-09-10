/*
 * BranchLengthOptimization.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../Network.hpp"

extern "C"
{
#include <libpll/pll_tree.h>
}

namespace netrax {
	double optimize_branches(Network& network, pllmod_treeinfo_t& fake_treeinfo, double lh_epsilon, double brlen_smooth_factor);
}
