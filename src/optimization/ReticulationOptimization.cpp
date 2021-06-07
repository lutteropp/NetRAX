
#include "ReticulationOptimization.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../RaxmlWrapper.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../utils.hpp"

#include "../likelihood/ComplexityScoring.hpp"

namespace netrax {

struct BrentBrprobParams {
  AnnotatedNetwork *ann_network;
  size_t reticulation_index;
};

void setReticulationProb(AnnotatedNetwork &ann_network, size_t reticulation_idx,
                         double prob) {
  if (ann_network.reticulation_probs[reticulation_idx] == prob) {
    return;
  }
  assert(reticulation_idx < ann_network.network.num_reticulations());
  assert(prob >= ann_network.options.brprob_min &&
         prob <= ann_network.options.brprob_max);
  ann_network.reticulation_probs[reticulation_idx] = prob;
  ann_network.first_parent_logprobs[reticulation_idx] = log(prob);
  ann_network.second_parent_logprobs[reticulation_idx] = log(1.0 - prob);
  ann_network.cached_logl_valid = false;
  invalidateTreeLogprobs(ann_network, reticulation_idx);
}

static double brent_target_networks_prob(void *p, double x) {
  AnnotatedNetwork *ann_network = ((BrentBrprobParams *)p)->ann_network;
  size_t reticulation_idx = ((BrentBrprobParams *)p)->reticulation_index;
  setReticulationProb(*ann_network, reticulation_idx, x);
  return -1 * computeLoglikelihood(*ann_network, 1, 1);
}

double optimize_reticulation_linear_search(AnnotatedNetwork &ann_network,
                                           size_t reticulation_index) {
  double best_prob = ann_network.reticulation_probs[reticulation_index];
  double best_logl = computeLoglikelihood(ann_network);

  double step = 1.0 / 100;

  for (int i = 0; i <= 1 / step; ++i) {
    double mid = i * step;
    setReticulationProb(ann_network, reticulation_index, mid);
    double act_logl = computeLoglikelihood(ann_network);
    if (act_logl > best_logl) {
      best_logl = act_logl;
      best_prob = mid;
    }
  }

  setReticulationProb(ann_network, reticulation_index, best_prob);
  return best_prob;
}

double optimize_reticulation(AnnotatedNetwork &ann_network,
                             size_t reticulation_index) {
  assert(reticulation_index < ann_network.network.num_reticulations());
  // return optimize_reticulation_linear_search(ann_network,
  // reticulation_index);

  double min_brprob = ann_network.options.brprob_min;
  double max_brprob = ann_network.options.brprob_max;
  double tolerance = ann_network.options.tolerance;

  double start_logl = computeLoglikelihood(ann_network, 1, 1);

  double best_logl = start_logl;
  BrentBrprobParams params;
  params.ann_network = &ann_network;
  params.reticulation_index = reticulation_index;
  double old_brprob = ann_network.reticulation_probs[reticulation_index];

  assert(old_brprob >= min_brprob);
  assert(old_brprob <= max_brprob);

  double score = 0;
  double f2x;
  double new_brprob = pllmod_opt_minimize_brent(
      min_brprob, old_brprob, max_brprob, tolerance, &score, &f2x,
      (void *)&params, &brent_target_networks_prob);

  setReticulationProb(ann_network, reticulation_index, new_brprob);
  return computeLoglikelihood(ann_network, 1, 1);
}

double optimize_reticulations(AnnotatedNetwork &ann_network, int max_iters) {
  double act_logl = computeLoglikelihood(ann_network, 1, 1);
  int act_iters = 0;
  while (act_iters < max_iters) {
    double loop_logl = act_logl;
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
      loop_logl = optimize_reticulation(ann_network, i);
    }
    act_iters++;
    if (loop_logl == act_logl) {
      break;
    }
    act_logl = loop_logl;
  }
  return act_logl;
}

}  // namespace netrax
