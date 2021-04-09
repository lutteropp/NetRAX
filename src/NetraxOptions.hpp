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
#include "LikelihoodVariant.hpp"

namespace netrax {

enum class BrlenOptMethod {
    BRENT_NORMAL = 0,
    BRENT_REROOT = 1,
    BRENT_REROOT_SUMTABLE = 2,
    NEWTON_RAPHSON = 3
};

class NetraxOptions {
public:
    NetraxOptions() {
    }

    NetraxOptions(const std::string &start_network_file, const std::string &msa_file, bool use_repeats =
            false) {
        this->start_network_file = start_network_file;
        this->msa_file = msa_file;
        this->use_repeats = use_repeats;
    }

    LikelihoodVariant likelihood_variant = LikelihoodVariant::AVERAGE_DISPLAYED_TREES;

    bool optimize_brlen = true;
    bool optimize_model = true;
    bool use_repeats = false;

    bool score_only = false;
    bool extract_taxon_names = false;
    bool extract_displayed_trees = false;
    bool check_weird_network = false;
    bool generate_random_network_only = false;
    bool pretty_print_only = false;

    bool change_reticulation_probs_only = false;
    double overwritten_reticulation_prob = -1;

    bool network_distance_only = false;
    std::string first_network_path = "";
    std::string second_network_path = "";

    double scale_branches_only = 0.0;

    bool endless = false;
    int seed = 0;

    unsigned int max_reticulations = 32;

    unsigned int num_random_start_networks = 10;
    unsigned int num_parsimony_start_networks = 10;

    unsigned int timeout = 0; // maximum number of seconds to run the network search, value of zero will be ignored

    bool use_extreme_greedy = false;
    bool no_prefiltering = false;
    bool use_rspr_moves = false;
    bool use_rspr1_moves = false;
    bool full_arc_insertion = false;
    bool less_moves = false;
    unsigned int scrambling = 0;
    unsigned int scrambling_radius = 1;

    bool sim_anneal = false;
    double start_temperature = 100;

    int brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
    int brlen_opt_method = PLLMOD_OPT_BLO_NEWTON_FAST;
    double brlen_min = RAXML_BRLEN_MIN;
    double brlen_max = RAXML_BRLEN_MAX;
    double brprob_min = 0.0;
    double brprob_max = 1.0;
    double lh_epsilon = DEF_LH_EPSILON;
    double tolerance = DEF_LH_EPSILON; //RAXML_BRLEN_TOLERANCE;
    double brlen_smoothings = RAXML_BRLEN_SMOOTHINGS;

    BrlenOptMethod brlenOptMethod = BrlenOptMethod::NEWTON_RAPHSON;//BrlenOptMethod::BRENT_REROOT; //BrlenOptMethod::BRENT_REROOT_SUMTABLE; //BrlenOptMethod::BRENT_REROOT;//BrlenOptMethod::NEWTON_RAPHSON_REROOT;// BrlenOptMethod::BRENT_NORMAL;

    std::string msa_file = "";
    std::string model_file = "DNA";
    std::string start_network_file = "";
    std::string output_file = "";
};
}
