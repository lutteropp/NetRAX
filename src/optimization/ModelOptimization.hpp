/*
 * ModelOptimization.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once


extern "C"
{
#include <libpll/pll_tree.h>
}
#include "../Network.hpp"
#include <vector>

namespace netrax {
	void optimizeModel(Network& network, pllmod_treeinfo_t& fake_treeinfo, double lh_epsilon);
}
