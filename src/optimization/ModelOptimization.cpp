/*
 * ModelOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "ModelOptimization.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../Network.hpp"
#include <raxml-ng/constants.hpp>
#include <raxml-ng/common.h>
#include <libpll/pll_optimize.h>
#include <libpll/pll_tree.h>

#include "BranchLengthOptimization.hpp"

#include "../Options.hpp"

static const bool _check_lh_impr = true;

namespace netrax {

void assert_lh_improvement(double old_lh, double new_lh, const std::string& where) {
	if (_check_lh_impr && !(old_lh - new_lh < -new_lh * RAXML_LOGLH_TOLERANCE)) {
		throw std::runtime_error(
				(where.empty() ? "" : "[" + where + "] ") + "Worse log-likelihood after optimization!\n" + "Old: " + std::to_string(old_lh)
						+ "\n"
								"New: " + std::to_string(new_lh) + "\n" + "NOTE: You can disable this check with '--force model_lh_impr'");
	}
}

void optimize_params_all(Network& network, pllmod_treeinfo_t& fake_treeinfo, double lh_epsilon) {
	optimize_params(network, fake_treeinfo, PLLMOD_OPT_PARAM_ALL, lh_epsilon);
}

double optimizeModel(Network& network, pllmod_treeinfo_t& fake_treeinfo, double lh_epsilon) {
	double new_loglh = computeLoglikelihood(network, fake_treeinfo);
	double cur_loglh;
	do {
		cur_loglh = new_loglh;
		optimize_params_all(network, fake_treeinfo, lh_epsilon);
		new_loglh = computeLoglikelihood(network, fake_treeinfo);
	} while (new_loglh - cur_loglh > lh_epsilon);
	return new_loglh;
}

}
