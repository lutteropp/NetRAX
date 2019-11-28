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
			optimize_brlen(true), optimize_model(true), use_repeats(false), num_reticulations(0), brlen_linkage(
					PLLMOD_COMMON_BRLEN_SCALED), brlen_opt_method(PLLMOD_OPT_BLO_NEWTON_FAST), brlen_min(
			RAXML_BRLEN_MIN), brlen_max(RAXML_BRLEN_MAX), msa_file(""), network_file("") {
	}

	bool optimize_brlen;
	bool optimize_model;
	bool use_repeats;

	unsigned int num_reticulations;

	int brlen_linkage;
	int brlen_opt_method;
	double brlen_min;
	double brlen_max;

	std::string msa_file;
	std::string network_file;
};
}
