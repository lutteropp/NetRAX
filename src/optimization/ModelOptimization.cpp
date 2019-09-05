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

double optimize_params(Network& network, std::vector<PartitionInfo>& partitions, int params_to_optimize, double lh_epsilon) {
	assert(!pll_errno);

	double cur_loglh = loglh(), new_loglh = cur_loglh;

	/* optimize SUBSTITUTION RATES */
	if (params_to_optimize & PLLMOD_OPT_PARAM_SUBST_RATES) {
		new_loglh = -1 * pllmod_algo_opt_subst_rates_treeinfo(_pll_treeinfo, 0,
		PLLMOD_OPT_MIN_SUBST_RATE,
		PLLMOD_OPT_MAX_SUBST_RATE,
		RAXML_BFGS_FACTOR,
		RAXML_PARAM_EPSILON);

		LOG_DEBUG << "\t - after rates: logLH = " << new_loglh << endl;

		libpll_check_error("ERROR in substitution rates optimization");
		assert_lh_improvement(cur_loglh, new_loglh, "RATES");
		cur_loglh = new_loglh;
	}

	/* optimize BASE FREQS */
	if (params_to_optimize & PLLMOD_OPT_PARAM_FREQUENCIES) {
		new_loglh = -1 * pllmod_algo_opt_frequencies_treeinfo(_pll_treeinfo, 0,
		PLLMOD_OPT_MIN_FREQ,
		PLLMOD_OPT_MAX_FREQ,
		RAXML_BFGS_FACTOR,
		RAXML_PARAM_EPSILON);

		LOG_DEBUG << "\t - after freqs: logLH = " << new_loglh << endl;

		libpll_check_error("ERROR in base frequencies optimization");
		assert_lh_improvement(cur_loglh, new_loglh, "FREQS");
		cur_loglh = new_loglh;
	}

	// TODO: co-optimization of PINV and ALPHA, mb with multiple starting points
	if (0 && (params_to_optimize & PLLMOD_OPT_PARAM_ALPHA) && (params_to_optimize & PLLMOD_OPT_PARAM_PINV)) {
		new_loglh = -1 * pllmod_algo_opt_alpha_pinv_treeinfo(_pll_treeinfo, 0,
		PLLMOD_OPT_MIN_ALPHA,
		PLLMOD_OPT_MAX_ALPHA,
		PLLMOD_OPT_MIN_PINV,
		PLLMOD_OPT_MAX_PINV,
		RAXML_BFGS_FACTOR,
		RAXML_PARAM_EPSILON);

		LOG_DEBUG << "\t - after a+i  : logLH = " << new_loglh << endl;

		libpll_check_error("ERROR in alpha/p-inv parameter optimization");
		assert_lh_improvement(cur_loglh, new_loglh, "ALPHA+PINV");
		cur_loglh = new_loglh;
	} else {
		/* optimize ALPHA */
		if (params_to_optimize & PLLMOD_OPT_PARAM_ALPHA) {
			new_loglh = -1 * pllmod_algo_opt_onedim_treeinfo(_pll_treeinfo,
			PLLMOD_OPT_PARAM_ALPHA,
			PLLMOD_OPT_MIN_ALPHA,
			PLLMOD_OPT_MAX_ALPHA,
			RAXML_PARAM_EPSILON);

			LOG_DEBUG << "\t - after alpha: logLH = " << new_loglh << endl;

			libpll_check_error("ERROR in alpha parameter optimization");
			assert_lh_improvement(cur_loglh, new_loglh, "ALPHA");
			cur_loglh = new_loglh;
		}

		/* optimize PINV */
		if (params_to_optimize & PLLMOD_OPT_PARAM_PINV) {
			new_loglh = -1 * pllmod_algo_opt_onedim_treeinfo(_pll_treeinfo,
			PLLMOD_OPT_PARAM_PINV,
			PLLMOD_OPT_MIN_PINV,
			PLLMOD_OPT_MAX_PINV,
			RAXML_PARAM_EPSILON);

			LOG_DEBUG << "\t - after p-inv: logLH = " << new_loglh << endl;

			libpll_check_error("ERROR in p-inv optimization");
			assert_lh_improvement(cur_loglh, new_loglh, "PINV");
			cur_loglh = new_loglh;
		}
	}

	/* optimize FREE RATES and WEIGHTS */
	if (params_to_optimize & PLLMOD_OPT_PARAM_FREE_RATES) {
		new_loglh = -1 * pllmod_algo_opt_rates_weights_treeinfo(_pll_treeinfo,
		RAXML_FREERATE_MIN,
		RAXML_FREERATE_MAX,
		RAXML_BFGS_FACTOR,
		RAXML_PARAM_EPSILON);

		/* normalize scalers and scale the branches accordingly */
		if (_pll_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED && _pll_treeinfo->partition_count > 1)
			pllmod_treeinfo_normalize_brlen_scalers (_pll_treeinfo);

		LOG_DEBUG << "\t - after freeR: logLH = " << new_loglh << endl;
		//    LOG_DEBUG << "\t - after freeR/crosscheck: logLH = " << loglh() << endl;

		libpll_check_error("ERROR in FreeRate rates/weights optimization");
		assert_lh_improvement(cur_loglh, new_loglh, "FREE RATES");
		cur_loglh = new_loglh;
	}

	if (params_to_optimize & PLLMOD_OPT_PARAM_BRANCHES_ITERATIVE) {
		new_loglh = optimize_branches(lh_epsilon, 0.25);

		assert_lh_improvement(cur_loglh, new_loglh, "BRLEN");
		cur_loglh = new_loglh;
	}

	return new_loglh;
}

void optimize_params_all(Network& network, std::vector<PartitionInfo>& partitions, double lh_epsilon) {
	optimize_params(network, partitions, PLLMOD_OPT_PARAM_ALL, lh_epsilon);
}

double optimizeModel(Network& network, std::vector<PartitionInfo>& partitions, double lh_epsilon) {
	double new_loglh = computeLoglikelihood(network, partitions);
	double cur_loglh;
	do {
		cur_loglh = new_loglh;
		optimize_params_all(network, partitions, lh_epsilon);
		new_loglh = computeLoglikelihood(network, partitions);
	} while (new_loglh - cur_loglh > lh_epsilon);
	return new_loglh;
}

}
