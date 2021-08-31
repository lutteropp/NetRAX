/*
 * Options.hpp
 *
 *  Created on: Sep 2, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <raxml-ng/common.h>
#include <raxml-ng/constants.hpp>
#include <string>
#include "likelihood/LikelihoodVariant.hpp"

namespace netrax {

enum class BrlenOptMethod {
  BRENT_NORMAL = 0,
  BRENT_REROOT = 1,
  NEWTON_RAPHSON = 2
};

class NetraxOptions {
 public:
  NetraxOptions() {}

  NetraxOptions(const std::string &start_network_file,
                const std::string &msa_file, bool use_repeats = false) {
    this->start_network_file = start_network_file;
    this->msa_file = msa_file;
    this->use_repeats = use_repeats;
  }

  size_t num_ranks = 1;

  LikelihoodVariant likelihood_variant = LikelihoodVariant::BEST_DISPLAYED_TREE;

  bool optimize_brlen = true;
  bool optimize_model = true;
  bool use_repeats = false;

  bool score_only = false;
  bool extract_taxon_names = false;
  bool extract_displayed_trees = false;
  bool check_weird_network = false;
  bool generate_random_network_only = false;
  bool pretty_print_only = false;

  int generate_n_parsimony = 0;
  int generate_n_random = 0;

  bool change_reticulation_probs_only = false;
  double overwritten_reticulation_prob = -1;

  bool network_distance_only = false;
  std::string first_network_path = "";
  std::string second_network_path = "";

  double scale_branches_only = 0.0;

  bool endless = false;
  long seed = 0;

  unsigned int max_reticulations = 32;

  unsigned int timeout = 0;  // maximum number of seconds to run the network
                             // search, value of zero will be ignored

  bool extreme_greedy = false;
  bool reorder_candidates = false;

  bool judge_only = false;

  size_t prefilter_keep = 60;
  size_t rank_keep = 20;

  bool no_prefiltering = false;
  bool no_rnni_moves = false;
  bool no_rspr_moves = false;
  bool no_arc_removal_moves = false;
  bool no_arc_insertion_moves = false;
  bool enforce_extra_search = false;
  unsigned int scrambling = 0;
  unsigned int scrambling_radius = 1;
  bool scrambling_only = false;

  bool good_start = false;

  int step_size = 5;

  bool prefilter_greedy = false;

  int max_rearrangement_distance = 25;

  bool sim_anneal = false;
  double start_temperature = 100;

  int brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
  int brlen_opt_method = PLLMOD_OPT_BLO_NEWTON_FAST;
  double brlen_min = RAXML_BRLEN_MIN;
  double brlen_max = RAXML_BRLEN_MAX;
  double brprob_min = 1E-6;
  double brprob_max = 1.0 - 1E-6;
  double lh_epsilon = DEF_LH_EPSILON;
  double tolerance = DEF_LH_EPSILON;  // RAXML_BRLEN_TOLERANCE;
  double brlen_smoothings = RAXML_BRLEN_SMOOTHINGS;

  double min_interesting_tree_logprob = log(1E-6);

  bool run_single_threaded = false;

  bool slow_mode = false;

  bool save_memory = false;

  bool randomize_candidates = false;

  bool reticulation_after_reticulation = false;

  int retry = 0;

  std::string true_network_path;

  BrlenOptMethod brlenOptMethod =
      BrlenOptMethod::NEWTON_RAPHSON;  // BrlenOptMethod::BRENT_REROOT;//BrlenOptMethod::NEWTON_RAPHSON_REROOT;//
                                       // BrlenOptMethod::BRENT_NORMAL;
  // BrlenOptMethod brlenOptMethod = BrlenOptMethod::BRENT_NORMAL;

  LoadBalancing load_balance_method =
      LoadBalancing::benoit;  // LoadBalancing::naive;

  bool no_elbow_method = false;

  int max_better_candidates = std::numeric_limits<int>::max();

  std::string msa_file = "";
  std::string model_file = "DNA";
  std::string start_network_file = "";
  std::string output_file = "";
};

inline bool no_parallelization_needed(const NetraxOptions &netraxOptions) {
  if (netraxOptions.pretty_print_only) {
    return true;
  } else if (netraxOptions.extract_displayed_trees) {
    return true;
  } else if (netraxOptions.check_weird_network) {
    return true;
  } else if (netraxOptions.generate_random_network_only) {
    return true;
  } else if (netraxOptions.scale_branches_only != 0.0) {
    return true;
  } else if (netraxOptions.change_reticulation_probs_only) {
    return true;
  } else if (netraxOptions.network_distance_only) {
    return true;
  } else if (netraxOptions.run_single_threaded) {
    return true;
  }
  return false;
}

}  // namespace netrax
