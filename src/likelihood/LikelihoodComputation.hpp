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

double computeLoglikelihood(Network& network, pllmod_treeinfo_t& fake_treeinfo);

}
