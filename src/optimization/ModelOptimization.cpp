#include "ModelOptimization.hpp"

#include "../graph/AnnotatedNetwork.hpp"
#include "../likelihood/LikelihoodComputation.hpp"

namespace netrax {

bool assert_lh_improvement(double old_lh, double new_lh, const std::string& where)
{
  if (!(old_lh - new_lh < -new_lh * RAXML_LOGLH_TOLERANCE))
  {
    throw std::runtime_error((where.empty() ? "" : "[" + where + "] ") +
                        "Worse log-likelihood after optimization!\n" +
                        "Old: " + to_string(old_lh) + "\n"
                        "New: " + to_string(new_lh) + "\n" +
                        "NOTE: You can disable this check with '--force model_lh_impr'");
  }
  return true;
}

double optimize_params(AnnotatedNetwork& ann_network, double lh_epsilon)
{
  assert(!pll_errno);

  double
    cur_loglh = computeLoglikelihood(ann_network),
    new_loglh = cur_loglh;

  /* optimize SUBSTITUTION RATES */
    new_loglh = -1 * pllmod_algo_opt_subst_rates_treeinfo(ann_network.fake_treeinfo,
                                                          0,
                                                          PLLMOD_OPT_MIN_SUBST_RATE,
                                                          PLLMOD_OPT_MAX_SUBST_RATE,
                                                          RAXML_BFGS_FACTOR,
                                                          RAXML_PARAM_EPSILON);

    libpll_check_error("ERROR in substitution rates optimization");
    assert(assert_lh_improvement(cur_loglh, new_loglh, "RATES"));
    cur_loglh = new_loglh;

    //std::cout << "thread " << ParallelContext::local_proc_id() << " survived substitution rates" << "\n";

  /* optimize BASE FREQS */
    new_loglh = -1 * pllmod_algo_opt_frequencies_treeinfo(ann_network.fake_treeinfo,
                                                          0,
                                                          PLLMOD_OPT_MIN_FREQ,
                                                          PLLMOD_OPT_MAX_FREQ,
                                                          RAXML_BFGS_FACTOR,
                                                          RAXML_PARAM_EPSILON);

    libpll_check_error("ERROR in base frequencies optimization");
    assert(assert_lh_improvement(cur_loglh, new_loglh, "FREQS"));
    cur_loglh = new_loglh;

    //std::cout << "thread " << ParallelContext::local_proc_id() << " survived base freqs" << "\n";

    /* optimize ALPHA */
      new_loglh = -1 * pllmod_algo_opt_onedim_treeinfo(ann_network.fake_treeinfo,
                                                        PLLMOD_OPT_PARAM_ALPHA,
                                                        PLLMOD_OPT_MIN_ALPHA,
                                                        PLLMOD_OPT_MAX_ALPHA,
                                                        RAXML_PARAM_EPSILON);

     libpll_check_error("ERROR in alpha parameter optimization");
     assert(assert_lh_improvement(cur_loglh, new_loglh, "ALPHA"));
     cur_loglh = new_loglh;

    //std::cout << "thread " << ParallelContext::local_proc_id() << " survived alpha" << "\n";

    /* optimize PINV */
      new_loglh = -1 * pllmod_algo_opt_onedim_treeinfo(ann_network.fake_treeinfo,
                                                        PLLMOD_OPT_PARAM_PINV,
                                                        PLLMOD_OPT_MIN_PINV,
                                                        PLLMOD_OPT_MAX_PINV,
                                                        RAXML_PARAM_EPSILON);

      libpll_check_error("ERROR in p-inv optimization");
      assert(assert_lh_improvement(cur_loglh, new_loglh, "PINV"));
      cur_loglh = new_loglh;

    //std::cout << "thread " << ParallelContext::local_proc_id() << " survived pinv" << "\n";

  /* optimize FREE RATES and WEIGHTS */
    new_loglh = -1 * pllmod_algo_opt_rates_weights_treeinfo (ann_network.fake_treeinfo,
                                                            RAXML_FREERATE_MIN,
                                                            RAXML_FREERATE_MAX,
                                                            ann_network.options.brlen_min,
                                                            ann_network.options.brlen_max,
                                                            RAXML_BFGS_FACTOR,
                                                            RAXML_PARAM_EPSILON);

    libpll_check_error("ERROR in FreeRate rates/weights optimization");
    assert(assert_lh_improvement(cur_loglh, new_loglh, "FREE RATES"));
    cur_loglh = new_loglh;

    //std::cout << "thread " << ParallelContext::local_proc_id() << " survived rates weights" << "\n";

  return new_loglh;
}

/**
 * Re-infers the likelihood model parameters of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeModel(AnnotatedNetwork &ann_network, bool silent) {
    //assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    if (!silent) std::cout << "BIC score before model optimization: " << old_score << "\n";

    optimize_params(ann_network, ann_network.options.lh_epsilon);

    //assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double new_score = scoreNetwork(ann_network);
    if (!silent) std::cout << "BIC score after model optimization: " << new_score << "\n";
    assert(new_score <= old_score);
}

}