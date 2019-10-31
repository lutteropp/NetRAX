/*
 * BranchLengthOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "BranchLengthOptimization.hpp"

#include <stdexcept>

#include "../graph/Common.hpp"

namespace netrax {
double optimize_branches(Network &network, pllmod_treeinfo_t &fake_treeinfo, double min_brlen, double max_brlen,
		double lh_epsilon, int max_iters, int opt_method, int radius) {
	// for now, optimize branches on each of the displayed trees, exported as a pll_utree_t data structure.
	// Keep track of which branch in the exported pll_utree_t corresponds to which branch of the network data structure.
	// Run br-length optimization on each of the displayed trees.
	// For each branch in the network, collect the optimized branch lengths from the displayed trees.
	// Print the optimized displayed tree branch lengths (as well as displayed tree prob) for each branch in the network to a file.
	// for now, set the network brlens to a weighted average of the displayed-tree-brlens (weighted by tree probability)
	// also print the new network brlens.
	// Do some plots.
	throw std::runtime_error("Not implemented yet");
}
}
