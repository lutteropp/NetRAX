/*
 * BranchLengthOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "BranchLengthOptimization.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../DebugPrintFunctions.hpp"
#include "../RaxmlWrapper.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../helper/Helper.hpp"
#include "../io/NetworkIO.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../utils.hpp"

#include "../graph/NodeDisplayedTreeData.hpp"
#include "../likelihood/LikelihoodDerivatives.hpp"
#include "../likelihood/VirtualRerooting.hpp"

#include "../colormod.h"

#include "NetworkState.hpp"

namespace netrax {

std::vector<DisplayedTreeData> extractOldTrees(AnnotatedNetwork &ann_network,
                                               Node *virtual_root) {
  if (!clvValidCheck(ann_network, virtual_root->clv_index)) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << exportDebugInfo(ann_network) << "\n";
    }
    throw std::runtime_error(
        "Cannot reuse old displayed trees before the extractOldTrees step. For "
        "some reason, they are invalidated at the root node " +
        std::to_string(virtual_root->clv_index));
  }
  std::vector<DisplayedTreeData> oldTrees;
  NodeDisplayedTreeData &nodeTrees =
      ann_network.pernode_displayed_tree_data[virtual_root->clv_index];
  for (size_t i = 0; i < nodeTrees.num_active_displayed_trees; ++i) {
    DisplayedTreeData &tree = nodeTrees.displayed_trees[i];
    oldTrees.emplace_back(tree);
  }
  return oldTrees;
}

struct BrentBrlenParams {
  AnnotatedNetwork *ann_network;
  size_t pmatrix_index;
  size_t partition_index;
  std::vector<DisplayedTreeData> *oldTrees = nullptr;
  BrlenOptMethod brlenOptMethod;
};

static double brent_target_networks(void *p, double x) {
  AnnotatedNetwork *ann_network = ((BrentBrlenParams *)p)->ann_network;
  size_t pmatrix_index = ((BrentBrlenParams *)p)->pmatrix_index;
  size_t partition_index = ((BrentBrlenParams *)p)->partition_index;
  std::vector<DisplayedTreeData> *oldTrees = ((BrentBrlenParams *)p)->oldTrees;
  BrlenOptMethod brlenOptMethod = ((BrentBrlenParams *)p)->brlenOptMethod;

  double old_x;
  if (ann_network->fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    old_x = ann_network->fake_treeinfo
                ->branch_lengths[partition_index][pmatrix_index];
  } else {
    old_x = ann_network->fake_treeinfo->linked_branch_lengths[pmatrix_index];
  }
  double score;
  if (old_x == x) {
    if (brlenOptMethod == BrlenOptMethod::BRENT_REROOT) {
      score = -1 * computeLoglikelihoodBrlenOpt(*ann_network, *oldTrees,
                                                pmatrix_index, 1);
    } else {  // BRENT_NORMAL
      score = -1 * computeLoglikelihood(*ann_network);
    }
  } else {
    if (ann_network->fake_treeinfo->brlen_linkage ==
        PLLMOD_COMMON_BRLEN_UNLINKED) {
      ann_network->fake_treeinfo
          ->branch_lengths[partition_index][pmatrix_index] = x;
    } else {
      ann_network->fake_treeinfo->linked_branch_lengths[pmatrix_index] = x;
    }

    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
      invalidPmatrixIndexOnly(*ann_network, pmatrix_index);
    } else {
      invalidatePmatrixIndex(*ann_network, pmatrix_index);
    }

    if (brlenOptMethod == BrlenOptMethod::BRENT_REROOT) {
      score = -1 * computeLoglikelihoodBrlenOpt(*ann_network, *oldTrees,
                                                pmatrix_index, 1);
    } else {  // BRENT_NORMAL
      score = -1 * computeLoglikelihood(*ann_network, 1, 1);
    }
  }

  return score;
}

double optimize_branch_brent(AnnotatedNetwork &ann_network,
                             std::vector<DisplayedTreeData> &oldTrees,
                             std::vector<std::vector<SumtableInfo>> &sumtables,
                             size_t pmatrix_index, size_t partition_index,
                             BrlenOptMethod brlenOptMethod) {
  assert(brlenOptMethod == BrlenOptMethod::BRENT_NORMAL ||
         brlenOptMethod == BrlenOptMethod::BRENT_REROOT);
  double old_brlen;
  if (ann_network.fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    old_brlen = ann_network.fake_treeinfo
                    ->branch_lengths[partition_index][pmatrix_index];
  } else {
    old_brlen = ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index];
  }
  assert(old_brlen >= ann_network.options.brlen_min);
  assert(old_brlen <= ann_network.options.brlen_max);

  BrentBrlenParams params;
  params.ann_network = &ann_network;
  params.pmatrix_index = pmatrix_index;
  params.partition_index = partition_index;
  params.oldTrees = &oldTrees;
  params.brlenOptMethod = brlenOptMethod;

  double min_brlen = ann_network.options.brlen_min;
  double max_brlen = ann_network.options.brlen_max;
  double tolerance = ann_network.options.tolerance;

  // Do Brent's method to find a better branch length
  double score = 0;
  double f2x;
  double new_brlen = pllmod_opt_minimize_brent(
      min_brlen, old_brlen, max_brlen, tolerance, &score, &f2x, (void *)&params,
      &brent_target_networks);
  assert(new_brlen >= ann_network.options.brlen_min);
  assert(new_brlen <= ann_network.options.brlen_max);

  if (ann_network.fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index] =
        new_brlen;
  } else {
    ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index] = new_brlen;
  }
  invalidatePmatrixIndex(ann_network, pmatrix_index);

  return new_brlen;
}

struct NewtonBrlenParams {
  AnnotatedNetwork *ann_network;
  size_t pmatrix_index;
  size_t partition_index;
  std::vector<DisplayedTreeData> *oldTrees = nullptr;
  std::vector<std::vector<SumtableInfo>> *sumtables = nullptr;
  double new_brlen;
};

static void network_derivative_func_multi(void *parameters, double *proposal,
                                          double *df, double *ddf) {
  NewtonBrlenParams *params = (NewtonBrlenParams *)parameters;
  AnnotatedNetwork *ann_network = params->ann_network;

  if (ann_network->fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    ann_network->fake_treeinfo
        ->branch_lengths[params->partition_index][params->pmatrix_index] =
        params->new_brlen;
  } else {
    ann_network->fake_treeinfo->linked_branch_lengths[params->pmatrix_index] =
        params->new_brlen;
  }

  invalidPmatrixIndexOnly(*ann_network, params->pmatrix_index);
  LoglDerivatives logl_derivatives = computeLoglikelihoodDerivatives(
      *ann_network, *(params->sumtables), params->pmatrix_index);

  if (ann_network->fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    for (size_t p = 0; p < ann_network->fake_treeinfo->partition_count; ++p) {
      df[p] = logl_derivatives.partition_logl_prime[p];
      ddf[p] = logl_derivatives.partition_logl_prime_prime[p];
    }
  } else {
    *df = logl_derivatives.logl_prime;
    *ddf = logl_derivatives.logl_prime_prime;
  }
}

double optimize_branch_newton_raphson(
    AnnotatedNetwork &ann_network,
    std::vector<std::vector<SumtableInfo>> &sumtables,
    std::vector<DisplayedTreeData> &oldTrees, size_t pmatrix_index,
    size_t partition_index, unsigned int max_iters) {
  double old_brlen;
  if (ann_network.fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    old_brlen = ann_network.fake_treeinfo
                    ->branch_lengths[partition_index][pmatrix_index];
  } else {
    old_brlen = ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index];
  }

  assert(old_brlen >= ann_network.options.brlen_min);
  assert(old_brlen <= ann_network.options.brlen_max);

  double tolerance = (ann_network.options.brlen_min > 0)
                         ? ann_network.options.brlen_min / 10.0
                         : PLLMOD_OPT_TOL_BRANCH_LEN;

  NewtonBrlenParams params;
  params.ann_network = &ann_network;
  params.partition_index = partition_index;
  params.pmatrix_index = pmatrix_index;
  params.oldTrees = &oldTrees;
  params.sumtables = &sumtables;
  params.new_brlen = old_brlen;
  pllmod_opt_minimize_newton_multi(
      1, ann_network.options.brlen_min, &(params.new_brlen),
      ann_network.options.brlen_max, tolerance, max_iters, NULL, &params,
      network_derivative_func_multi);
  libpll_reset_error();

  double new_brlen = params.new_brlen;
  assert(new_brlen >= ann_network.options.brlen_min);
  assert(new_brlen <= ann_network.options.brlen_max);

  return new_brlen;
}

void add_neighbors_in_radius(AnnotatedNetwork &ann_network,
                             std::unordered_set<size_t> &candidates,
                             int pmatrix_index, int radius,
                             std::unordered_set<size_t> &seen) {
  if (seen.count(pmatrix_index) == 1 || radius == 0 ||
      candidates.size() == ann_network.network.num_branches()) {
    return;
  }
  seen.emplace(pmatrix_index);
  std::vector<Edge *> neighs = netrax::getAdjacentEdges(
      ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
  for (size_t i = 0; i < neighs.size(); ++i) {
    candidates.emplace(neighs[i]->pmatrix_index);
    add_neighbors_in_radius(ann_network, candidates, neighs[i]->pmatrix_index,
                            radius - 1, seen);
  }
}

void add_neighbors_in_radius(AnnotatedNetwork &ann_network,
                             std::unordered_set<size_t> &candidates,
                             int radius) {
  if (radius == 0) {
    return;
  }
  std::unordered_set<size_t> seen;
  for (size_t pmatrix_index : candidates) {
    add_neighbors_in_radius(ann_network, candidates, pmatrix_index, radius,
                            seen);
  }
}

void add_neighbors_in_radius(AnnotatedNetwork &ann_network,
                             std::unordered_set<size_t> &candidates,
                             size_t pmatrix_index, int radius) {
  if (radius == 0) {
    return;
  }
  std::unordered_set<size_t> seen;
  add_neighbors_in_radius(ann_network, candidates, pmatrix_index, radius, seen);
}

double optimize_branch(AnnotatedNetwork &ann_network,
                       std::vector<DisplayedTreeData> &oldTrees,
                       std::vector<std::vector<SumtableInfo>> &sumtables,
                       size_t pmatrix_index, size_t partition_index,
                       BrlenOptMethod brlenOptMethod, unsigned int max_iters) {
  ann_network.cached_logl_valid = false;

  double start_logl;
  if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
    start_logl =
        computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1);
  } else {
    start_logl = computeLoglikelihood(ann_network);
  }

  if (brlenOptMethod == BrlenOptMethod::BRENT_NORMAL ||
      brlenOptMethod == BrlenOptMethod::BRENT_REROOT) {
    optimize_branch_brent(ann_network, oldTrees, sumtables, pmatrix_index,
                          partition_index, brlenOptMethod);
  } else {  // BrlenOptMethod::NEWTON_RAPHSON_REROOT
    double old_brlen;
    if (ann_network.fake_treeinfo->brlen_linkage ==
        PLLMOD_COMMON_BRLEN_UNLINKED) {
      old_brlen = ann_network.fake_treeinfo
                      ->branch_lengths[partition_index][pmatrix_index];
    } else {
      old_brlen =
          ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index];
    }
    optimize_branch_newton_raphson(ann_network, sumtables, oldTrees,
                                   pmatrix_index, partition_index, max_iters);
    double new_logl =
        computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1);
    if (new_logl <
        start_logl) {  // this can happen in rare cases, if NR didn't converge.
                       // If it happens, reoad the old branch length.
      // std::cout << "reload old brlen\n";
      if (ann_network.fake_treeinfo->brlen_linkage ==
          PLLMOD_COMMON_BRLEN_UNLINKED) {
        ann_network.fake_treeinfo
            ->branch_lengths[partition_index][pmatrix_index] = old_brlen;
      } else {
        ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index] =
            old_brlen;
      }
      invalidPmatrixIndexOnly(ann_network, pmatrix_index);
    }
  }

  double best_logl;
  if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
    best_logl =
        computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1);
  } else {
    best_logl = computeLoglikelihood(ann_network);
  }
  assert(best_logl >= start_logl);

  return best_logl;
}

double optimize_branch(AnnotatedNetwork &ann_network, size_t pmatrix_index,
                       BrlenOptMethod brlenOptMethod, unsigned int max_iters) {
  assert(pmatrix_index < ann_network.network.num_branches());
  double old_logl = computeLoglikelihood(ann_network);
  assert(old_logl <= 0.0);
  // assert(computeLoglikelihood(ann_network, 0, 1) == old_logl);
  std::vector<DisplayedTreeData> oldTrees;
  std::vector<std::vector<SumtableInfo>> sumtables;

  // step 1: Do the virtual rerooting.
  if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
    oldTrees = extractOldTrees(ann_network, ann_network.network.root);
    Node *old_virtual_root = ann_network.network.root;
    Node *new_virtual_root = getSource(
        ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
      Node *new_virtual_root_back =
          getTarget(ann_network.network,
                    ann_network.network.edges_by_index[pmatrix_index]);
      ReticulationConfigSet restrictions =
          getRestrictionsActiveAliveBranch(ann_network, pmatrix_index);
      updateCLVsVirtualRerootTrees(ann_network, old_virtual_root,
                                   new_virtual_root, new_virtual_root_back,
                                   restrictions);
    }
    ann_network.cached_logl_valid = false;

    // Leaving out this check is dangerous...
    double brlenopt_logl =
        computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index);
    if (fabs(old_logl - brlenopt_logl >= 1E-3)) {
      if (ParallelContext::master_rank() && ParallelContext::master_thread) {
        std::cout << exportDebugInfo(ann_network) << "\n";
        ann_network.cached_logl_valid = false;
        computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1,
                                     true);
        std::cout << "old_logl: " << old_logl << "\n";
        std::cout << "brlenopt_logl: " << brlenopt_logl << "\n";
        std::cout << "problem occurred while optimizing branch "
                  << pmatrix_index << "\n";
      }
      ParallelContext::mpi_barrier();
      throw std::runtime_error(
          "Something went wrong when rerooting CLVs during brlen optimization");
    }

    if (brlenOptMethod == BrlenOptMethod::NEWTON_RAPHSON) {
      sumtables = computePartitionSumtables(ann_network, pmatrix_index);
    }
  }

  ann_network.fake_treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
  if (ann_network.fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      optimize_branch(ann_network, oldTrees, sumtables, pmatrix_index, p,
                      brlenOptMethod, max_iters);
    }
  } else {
    optimize_branch(ann_network, oldTrees, sumtables, pmatrix_index, 0,
                    brlenOptMethod,
                    max_iters);  // partition_idx will get ignored anyway
  }

  // restore the network root
  if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
    invalidatePmatrixIndex(ann_network, pmatrix_index);
  }

  double final_logl = computeLoglikelihood(ann_network);
  return final_logl;
}

double optimize_branches_internal(AnnotatedNetwork &ann_network, int max_iters,
                                  int max_iters_outside, int radius,
                                  std::unordered_set<size_t> candidates,
                                  bool restricted_total_iters) {
  for (size_t idx : candidates) {
    assert(idx < ann_network.network.num_branches());
  }
  double lh_epsilon = ann_network.options.lh_epsilon;
  double old_logl = computeLoglikelihood(ann_network, 1, 1);
  double start_logl = old_logl;
  std::vector<size_t> act_iters(ann_network.network.num_branches(), 0);
  BrlenOptMethod brlenOptMethod = ann_network.options.brlenOptMethod;

  /*if (ann_network.network.num_reticulations() == 0) {
  brlenOptMethod = BrlenOptMethod::NEWTON_RAPHSON_REROOT;
}*/

  Node *old_virtual_root = ann_network.network.root;

  size_t total_iters = 0;

  while (!candidates.empty()) {
    size_t pmatrix_index = *candidates.begin();
    candidates.erase(candidates.begin());
    // std::cout << "optimizing branch " << pmatrix_index << "\n";

    total_iters++;
    if (restricted_total_iters && total_iters >= max_iters_outside) {
      continue;
    }

    if (act_iters[pmatrix_index] >= max_iters_outside) {
      continue;
    }
    act_iters[pmatrix_index]++;

    double new_logl =
        optimize_branch(ann_network, pmatrix_index, brlenOptMethod, max_iters);

    /*if (new_logl - old_logl > lh_epsilon) { // add all neighbors of the branch
to the candidates add_neighbors_in_radius(ann_network, candidates,
pmatrix_index, 1);
}*/
    old_logl = new_logl;
    // old_virtual_root = new_virtual_root;
  }

  if ((old_logl < start_logl) && (fabs(old_logl - start_logl) >= 1E-3)) {
    std::cout << "old_logl: " << old_logl << "\n";
    std::cout << "start_logl: " << start_logl << "\n";
    throw std::runtime_error("Overall loglikelihood got worse");
  }
  assert((old_logl >= start_logl) || (fabs(old_logl - start_logl) < 1E-3));

  return old_logl;
}

ReticulationConfigSet decideInterestingTrees(
    AnnotatedNetwork &ann_network, std::unordered_set<size_t> &candidates) {
  // Find a set of trees such that all candidate branches are active in at least
  // one of the trees
  ReticulationConfigSet res;
  NodeDisplayedTreeData &ndtd =
      ann_network
          .pernode_displayed_tree_data[ann_network.network.root->clv_index];
  if (ndtd.num_active_displayed_trees <= 2) {
    return {};
  }

  unsigned int n_added = 0;
  // we need a tree with all zeros, and we need a tree with all ones
  ReticulationConfigSet allZero(ann_network.options.max_reticulations);
  ReticulationConfigSet allOne(ann_network.options.max_reticulations);
  std::vector<ReticulationState> vec(ann_network.options.max_reticulations,
                                     ReticulationState::DONT_CARE);
  allZero.configs.emplace_back(vec);
  allOne.configs.emplace_back(vec);
  for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
    allZero.configs[0][i] = ReticulationState::TAKE_FIRST_PARENT;
    allOne.configs[0][i] = ReticulationState::TAKE_SECOND_PARENT;
  }
  for (size_t i = 0; i < ndtd.num_active_displayed_trees; ++i) {
    if (reticulationConfigsCompatible(
            ndtd.displayed_trees[i].treeLoglData.reticulationChoices, allOne)) {
      addOrReticulationChoices(
          res, ndtd.displayed_trees[i].treeLoglData.reticulationChoices);
      n_added++;
    } else if (reticulationConfigsCompatible(
                   ndtd.displayed_trees[i].treeLoglData.reticulationChoices,
                   allZero)) {
      addOrReticulationChoices(
          res, ndtd.displayed_trees[i].treeLoglData.reticulationChoices);
      n_added++;
    }
  }

  /*for (size_t i = 0; i < ndtd.num_active_displayed_trees; ++i) {
    if (!reticulationConfigsCompatible(
            res, ndtd.displayed_trees[i].treeLoglData.reticulationChoices)) {
      addOrReticulationChoices(
          res, ndtd.displayed_trees[i].treeLoglData.reticulationChoices);
      n_added++;
    }
    if (n_added == ndtd.num_active_displayed_trees / 2) {
      break;
    }
  }*/
  if (n_added == ndtd.num_active_displayed_trees) {
    res.configs.clear();
  }

  return res;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters,
                         int max_iters_outside, int radius,
                         std::unordered_set<size_t> candidates,
                         bool restricted_total_iters) {
  /*double old_logl = computeLoglikelihood(ann_network);
  double new_logl;

  if (ann_network.network.num_reticulations() > 1) {
    NetworkState oldState = extract_network_state(ann_network);
    // try first optimizing the braanch on just a subset of displayed trees
    ann_network.clearInterestingTreeRestriction();
    ReticulationConfigSet rcs = decideInterestingTrees(ann_network, candidates);
    ann_network.addInterestingTreeRestriction(rcs);
    optimize_branches_internal(ann_network, max_iters, max_iters_outside,
                               radius, candidates, restricted_total_iters);
    ann_network.clearInterestingTreeRestriction();
    new_logl = computeLoglikelihood(ann_network);
    if (new_logl < old_logl) {
      apply_network_state(ann_network, oldState);
      new_logl = optimize_branches_internal(ann_network, max_iters,
                                            max_iters_outside, radius,
                                            candidates, restricted_total_iters);
    }
  } else {
    new_logl =
        optimize_branches_internal(ann_network, max_iters, max_iters_outside,
                                   radius, candidates, restricted_total_iters);
  }
  return new_logl;*/

  return optimize_branches_internal(ann_network, max_iters, max_iters_outside,
                                   radius, candidates, restricted_total_iters);
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters,
                         int max_iters_outside, int radius,
                         bool restricted_total_iters) {
  std::unordered_set<size_t> candidates;
  for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
    candidates.emplace(i);
  }
  return optimize_branches(ann_network, max_iters, max_iters_outside, radius,
                           candidates, restricted_total_iters);
}

double optimize_scalers(AnnotatedNetwork &ann_network, bool silent) {
  double old_score = scoreNetwork(ann_network);
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED &&
      ann_network.fake_treeinfo->partition_count > 1) {
    pllmod_algo_opt_brlen_scalers_treeinfo(
        ann_network.fake_treeinfo, RAXML_BRLEN_SCALER_MIN,
        RAXML_BRLEN_SCALER_MAX, ann_network.options.brlen_min,
        ann_network.options.brlen_max, RAXML_PARAM_EPSILON);
    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::master()) {
      std::cout << "BIC score after branch length scaler optimization: "
                << new_score << "\n";
    }
    assert(new_score <= old_score);
    return new_score;
  } else {
    return old_score;
  }
}

}  // namespace netrax
