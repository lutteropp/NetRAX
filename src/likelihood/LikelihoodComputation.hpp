/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

#include <libpll/pll.h>
#include <libpll/pll_tree.h>

#include "../Network.hpp"

namespace netrax {

void updateProbMatrices(Network& network, pllmod_treeinfo_t& fake_treeinfo, bool updateAll);
double computeLoglikelihood(Network& network, pllmod_treeinfo_t& fake_treeinfo);

}
