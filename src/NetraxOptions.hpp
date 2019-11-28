/*
 * Options.hpp
 *
 *  Created on: Sep 2, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <string>
#include <raxml-ng/constants.hpp>
#include <raxml-ng/common.h>

namespace netrax {

class NetraxOptions {
public:
	bool optimize_brlen = true;
	bool optimize_model = true;
	bool use_repeats = false;

	unsigned int num_reticulations = 0;

	int brlen_linkage = PLLMOD_COMMON_BRLEN_SCALED;
	int brlen_opt_method = PLLMOD_OPT_BLO_NEWTON_FAST;
	double brlen_min = RAXML_BRLEN_MIN;
	double brlen_max = RAXML_BRLEN_MAX;

	std::string msa_file = "";
	std::string network_file = "";
};
}
