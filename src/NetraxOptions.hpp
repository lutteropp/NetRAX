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
    NetraxOptions() {
    }

    NetraxOptions(const std::string &network_file, const std::string &msa_file, bool use_repeats = true) {
        this->network_file = network_file;
        this->msa_file = msa_file;
        this->use_repeats = use_repeats;
    }

    bool optimize_brlen = true;
    bool optimize_model = true;
    bool use_repeats = true;

    bool use_blobs = true;
    bool use_graycode = true;
    bool use_incremental = true;

    unsigned int max_reticulations = 20;

    int brlen_linkage = PLLMOD_COMMON_BRLEN_SCALED;
    int brlen_opt_method = PLLMOD_OPT_BLO_NEWTON_FAST;
    double brlen_min = RAXML_BRLEN_MIN;
    double brlen_max = RAXML_BRLEN_MAX;
    double lh_epsilon = DEF_LH_EPSILON;
    double tolerance = RAXML_BRLEN_TOLERANCE;
    double brlen_smoothings = RAXML_BRLEN_SMOOTHINGS;

    std::string msa_file = "";
    std::string network_file = "";
    std::string output_file = "";
};
}
