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
	NetraxOptions() :
			msa_file(""), network_file(""), num_reticulations(0), optimize_brlen(true), optimize_model(true), use_repeats(
					false), brlen_linkage(PLLMOD_COMMON_BRLEN_SCALED), brlen_opt_method(PLLMOD_OPT_BLO_NEWTON_FAST), brlen_min(
					RAXML_BRLEN_MIN), brlen_max(RAXML_BRLEN_MAX) {
	}

	std::string msa_file;
	std::string network_file;
	unsigned int num_reticulations;

	bool optimize_brlen;
	bool optimize_model;
	bool use_repeats;

	int brlen_linkage;
	int brlen_opt_method;
	double brlen_min;
	double brlen_max;
};
}
