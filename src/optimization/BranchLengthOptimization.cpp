/*
 * BranchLengthOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "BranchLengthOptimization.hpp"

#include <stdexcept>

#include "../Network.hpp"

namespace netrax {
double optimize_branches(Network& network, pllmod_treeinfo_t& fake_treeinfo, double min_brlen, double max_brlen, double lh_epsilon,
		int max_iters, int opt_method, int radius) {
	throw std::runtime_error("Not implemented yet");
}
}
