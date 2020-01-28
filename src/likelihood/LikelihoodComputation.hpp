/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

extern "C"
{
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "../graph/Common.hpp"
#include "../RaxmlWrapper.hpp"
#include <raxml-ng/TreeInfo.hpp>

namespace netrax {

std::vector<pll_operation_t> createOperations(Network& network, size_t treeIdx);

double displayed_tree_prob(Network &network, size_t tree_index, size_t partition_index);

double computeLoglikelihood(Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental, int update_pmatrices,
		bool update_reticulation_probs = false);

double computeLoglikelihoodNaiveUtree(RaxmlWrapper& wrapper, Network& network, int incremental, int update_pmatrices);
}
