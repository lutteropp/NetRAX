/*
 * BranchLengthOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "BranchLengthOptimization.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>

#include "../graph/NetworkFunctions.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../RaxmlWrapper.hpp"
#include "../utils.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../DebugPrintFunctions.hpp"

namespace netrax {

struct BrentBrlenParams {
    AnnotatedNetwork *ann_network;
    size_t pmatrix_index;
    size_t partition_index;
    std::vector<std::vector<TreeLoglData> >* oldTrees = nullptr;
    std::vector<std::vector<SumtableInfo> >* sumtables = nullptr;
    BrlenOptMethod brlenOptMethod;
};

static double brent_target_networks(void *p, double x) {
    AnnotatedNetwork *ann_network = ((BrentBrlenParams*) p)->ann_network;
    size_t pmatrix_index = ((BrentBrlenParams*) p)->pmatrix_index;
    size_t partition_index = ((BrentBrlenParams*) p)->partition_index;
    std::vector<std::vector<TreeLoglData>>* oldTrees = ((BrentBrlenParams*) p)->oldTrees;
    std::vector<std::vector<SumtableInfo>>* sumtables = ((BrentBrlenParams*) p)->sumtables;
    BrlenOptMethod brlenOptMethod = ((BrentBrlenParams*) p)->brlenOptMethod;

    double old_x = ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    double score;
    if (old_x == x) {
        if (brlenOptMethod == BrlenOptMethod::BRENT_REROOT_SUMTABLE) {
            score = -1 * computeLoglikelihoodFromSumtables(*ann_network, *sumtables, *oldTrees, pmatrix_index, 1, 1);
        } else if (brlenOptMethod == BrlenOptMethod::BRENT_REROOT) {
            score = -1 * computeLoglikelihoodBrlenOpt(*ann_network, *oldTrees, pmatrix_index, 1, 1);
        } else { // BRENT_NORMAL
            score = -1 * computeLoglikelihood(*ann_network);
        }
    } else {
        ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index] = x;
        invalidPmatrixIndexOnly(*ann_network, pmatrix_index);

        if (brlenOptMethod == BrlenOptMethod::BRENT_REROOT_SUMTABLE) {
            double sumtable_logl = computeLoglikelihoodFromSumtables(*ann_network, *sumtables, *oldTrees, pmatrix_index, 1, 1);
            double reroot_logl = computeLoglikelihoodBrlenOpt(*ann_network, *oldTrees, pmatrix_index, 1, 1);
            score = -1 * sumtable_logl;
            /*if (ann_network->network.num_reticulations() == 1) {
                std::cout << "logl from sumtables: " << sumtable_logl << "\n";
                std::cout << "logl from reroot: " << reroot_logl << "\n";
            }*/
            assert(fabs(sumtable_logl - reroot_logl) < 1E-3);
        }
        if (brlenOptMethod == BrlenOptMethod::BRENT_REROOT) {
            score = -1 * computeLoglikelihoodBrlenOpt(*ann_network, *oldTrees, pmatrix_index, 1, 1);
        } else { // BRENT_NORMAL
            score = -1 * computeLoglikelihood(*ann_network, 1, 1);
        }
    }

    return score;
}

double optimize_branch_brent(AnnotatedNetwork &ann_network, std::vector<std::vector<TreeLoglData> >& oldTrees, std::vector<std::vector<SumtableInfo> >& sumtables, size_t pmatrix_index, size_t partition_index, BrlenOptMethod brlenOptMethod) {
    assert(brlenOptMethod == BrlenOptMethod::BRENT_NORMAL || brlenOptMethod == BrlenOptMethod::BRENT_REROOT || brlenOptMethod == BrlenOptMethod::BRENT_REROOT_SUMTABLE);
    double old_brlen = ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    assert(old_brlen >= ann_network.options.brlen_min);
    assert(old_brlen <= ann_network.options.brlen_max);

    BrentBrlenParams params;
    params.ann_network = &ann_network;
    params.pmatrix_index = pmatrix_index;
    params.partition_index = partition_index;
    params.oldTrees = &oldTrees;
    params.sumtables = &sumtables;
    params.brlenOptMethod = brlenOptMethod;

    double min_brlen = ann_network.options.brlen_min;
    double max_brlen = ann_network.options.brlen_max;
    double tolerance = ann_network.options.tolerance;

    // Do Brent's method to find a better branch length
    double score = 0;
    double f2x;
    double new_brlen = pllmod_opt_minimize_brent(min_brlen, old_brlen, max_brlen, tolerance, &score,
            &f2x, (void*) &params, &brent_target_networks);
    assert(new_brlen >= ann_network.options.brlen_min);
    assert(new_brlen <= ann_network.options.brlen_max);

    return new_brlen;
}


struct NewtonBrlenParams
{
    AnnotatedNetwork *ann_network;
    size_t pmatrix_index;
    size_t partition_index;
    std::vector<std::vector<TreeLoglData> >* oldTrees = nullptr;
    std::vector<std::vector<SumtableInfo> >* sumtables = nullptr;
    double new_brlen;
};


static void network_derivative_func_multi (void * parameters, double * proposal,
                                         double *df, double *ddf)
{
  NewtonBrlenParams * params =
                                (NewtonBrlenParams *) parameters;
  AnnotatedNetwork* ann_network = params->ann_network;

  //std::cout << "proposing brlen: " << params->new_brlen << "\n";
  ann_network->fake_treeinfo->branch_lengths[params->partition_index][params->pmatrix_index] = params->new_brlen;
  invalidPmatrixIndexOnly(*ann_network, params->pmatrix_index);
  LoglDerivatives logl_derivatives = computeLoglikelihoodDerivatives(*ann_network, *(params->sumtables), *(params->oldTrees), params->pmatrix_index);

  int unlinked = (ann_network->options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) ? 1 : 0;

  if (unlinked) {
    for (size_t p = 0; p < ann_network->fake_treeinfo->partition_count; ++p) {
      df[p] = logl_derivatives.partition_logl_prime[p];
      ddf[p] = logl_derivatives.partition_logl_prime_prime[p];
    }
  } else {
    *df = logl_derivatives.logl_prime;
    *ddf = logl_derivatives.logl_prime_prime;
  }
}

double search_newton_raphson(AnnotatedNetwork& ann_network, std::vector<std::vector<SumtableInfo> >& sumtables, size_t pmatrix_index, size_t partition_index, unsigned int max_iters) {
    double old_brlen = ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    //LoglDerivatives logl_derivatives = computeLoglikelihoodDerivatives(ann_network, sumtables, oldTrees, pmatrix_index);

    double tolerance = (ann_network.options.brlen_min > 0) ? ann_network.options.brlen_min /10.0: PLLMOD_OPT_TOL_BRANCH_LEN;

    double xguess = old_brlen;
    /*for (size_t it = 0; it < max_iters; ++it) {
        xguess -= logl_derivatives.quotient.toDouble();
        std::cout << "proposing brlen: " << xguess << "\n";
        ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index] = xguess;
        invalidPmatrixIndexOnly(ann_network, pmatrix_index);
        logl_derivatives = computeLoglikelihoodDerivatives(ann_network, sumtables, pmatrix_index);


        if (fabs(xguess - logl_derivatives.quotient.toDouble()) < tolerance) {
            // converged
            break;
        }
    }
    return xguess;*/
    throw std::runtime_error("not implemented yet");
}

double optimize_branch_newton_raphson(AnnotatedNetwork &ann_network, std::vector<std::vector<SumtableInfo> >& sumtables, std::vector<std::vector<TreeLoglData> >& oldTrees, size_t pmatrix_index, size_t partition_index, BrlenOptMethod brlenOptMethod, unsigned int max_iters) {
    assert(brlenOptMethod == BrlenOptMethod::NEWTON_RAPHSON);

    double old_brlen = ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    assert(old_brlen >= ann_network.options.brlen_min);
    assert(old_brlen <= ann_network.options.brlen_max);

    double tolerance = (ann_network.options.brlen_min > 0) ? ann_network.options.brlen_min /10.0: PLLMOD_OPT_TOL_BRANCH_LEN;

    //double new_brlen = search_newton_raphson(ann_network, sumtables, pmatrix_index, partition_index, max_iters);

    //std::cout << "MAX_ITERS: \n" << max_iters << "\n";

    NewtonBrlenParams params;
    params.ann_network = &ann_network;
    params.partition_index = partition_index;
    params.pmatrix_index = pmatrix_index;
    params.oldTrees = &oldTrees;
    params.sumtables = &sumtables;
    params.new_brlen = old_brlen;
    pllmod_opt_minimize_newton_multi(1,
                                    ann_network.options.brlen_min,
                                    &(params.new_brlen),
                                    ann_network.options.brlen_max,
                                    tolerance,
                                    max_iters,
                                    NULL,
                                    &params,
                                    network_derivative_func_multi);
    if (pll_errno) {
        //std::cout << pll_errmsg << "\n";
    }
    libpll_reset_error();
    //assert(!pll_errno);

    double new_brlen = params.new_brlen;
    assert(new_brlen >= ann_network.options.brlen_min);
    assert(new_brlen <= ann_network.options.brlen_max);

    //std::cout << "old brlen: " << old_brlen << "\n";
    //std::cout << "new brlen: " << new_brlen << "\n";

    //throw std::runtime_error("I WANT TO QUIT HERE");

    return new_brlen;
}

void add_neighbors_in_radius(AnnotatedNetwork& ann_network, std::unordered_set<size_t>& candidates, int pmatrix_index, int radius, std::unordered_set<size_t> &seen) {
    if (seen.count(pmatrix_index) ==1 || radius == 0 || candidates.size() == ann_network.network.num_branches()) {
        return;
    }
    seen.emplace(pmatrix_index);
    std::vector<Edge*> neighs = netrax::getAdjacentEdges(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    for (size_t i = 0; i < neighs.size(); ++i) {
        candidates.emplace(neighs[i]->pmatrix_index);
        add_neighbors_in_radius(ann_network, candidates, neighs[i]->pmatrix_index, radius - 1, seen);
    }
}

void add_neighbors_in_radius(AnnotatedNetwork& ann_network, std::unordered_set<size_t>& candidates, int radius) {
    if (radius == 0) {
        return;
    }
    std::unordered_set<size_t> seen;
    for (size_t pmatrix_index : candidates) {
        add_neighbors_in_radius(ann_network, candidates, pmatrix_index, radius, seen);
    }
}

void add_neighbors_in_radius(AnnotatedNetwork& ann_network, std::unordered_set<size_t>& candidates, size_t pmatrix_index, int radius) {
    if (radius == 0) {
        return;
    }
    std::unordered_set<size_t> seen;
    add_neighbors_in_radius(ann_network, candidates, pmatrix_index, radius, seen);
}

double optimize_branch(AnnotatedNetwork &ann_network, std::vector<std::vector<TreeLoglData> >& oldTrees, std::vector<std::vector<SumtableInfo> >& sumtables, size_t pmatrix_index, size_t partition_index, BrlenOptMethod brlenOptMethod, unsigned int max_iters) {
    ann_network.cached_logl_valid = false;

    double start_logl;
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        start_logl = computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1, 1);
    } else {
        start_logl = computeLoglikelihood(ann_network);
    }

    if (brlenOptMethod == BrlenOptMethod::BRENT_NORMAL || brlenOptMethod == BrlenOptMethod::BRENT_REROOT || brlenOptMethod == BrlenOptMethod::BRENT_REROOT_SUMTABLE) {
        optimize_branch_brent(ann_network, oldTrees, sumtables, pmatrix_index, partition_index, brlenOptMethod);
    } else { // BrlenOptMethod::NEWTON_RAPHSON_REROOT
        //std::cout << "\nStarting with network logl: " << start_logl << "\n";
        double old_brlen = ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
        optimize_branch_newton_raphson(ann_network, sumtables, oldTrees, pmatrix_index, partition_index, brlenOptMethod, max_iters);
        double new_logl = computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1, 1);
        if (new_logl < start_logl) { // this can happen in rare cases, if NR didn't converge. If it happens, reoad the old branch length.
            //std::cout << "reload old brlen\n";
            ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index] = old_brlen;
            invalidPmatrixIndexOnly(ann_network, pmatrix_index);
        } /*else if (new_logl > start_logl && ann_network.network.num_reticulations() > 0) {
            std::cout << "actually found a better brlen with NR\n";
        }*/
    }

    double best_logl;
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        best_logl = computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1, 1);
    } else {
        best_logl = computeLoglikelihood(ann_network);
    }
    //std::cout << "start_logl: " << start_logl << "\n";
    //std::cout << "best_logl: " << best_logl << "\n";
    assert(best_logl >= start_logl);

    return best_logl;
}

double optimize_branch(AnnotatedNetwork &ann_network, size_t pmatrix_index, BrlenOptMethod brlenOptMethod, unsigned int max_iters) {
    double old_logl = computeLoglikelihood(ann_network);
    std::vector<std::vector<TreeLoglData> > oldTrees;
    std::vector<std::vector<SumtableInfo> > sumtables;

    // step 1: Do the virtual rerooting.
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        oldTrees = extractOldTrees(ann_network, ann_network.network.root);
        Node* old_virtual_root = ann_network.network.root;
        Node* new_virtual_root = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
        if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
            Node* new_virtual_root_back = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
            updateCLVsVirtualRerootTrees(ann_network, old_virtual_root, new_virtual_root, new_virtual_root_back);
        }
        ann_network.cached_logl_valid = false;
        assert(fabs(old_logl - computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index)) < 1E-3);

        if (brlenOptMethod == BrlenOptMethod::NEWTON_RAPHSON || brlenOptMethod == BrlenOptMethod::BRENT_REROOT_SUMTABLE) {
            sumtables = computePartitionSumtables(ann_network, pmatrix_index);
        }
    }

    size_t n_partitions = 1;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }

    ann_network.fake_treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    for (size_t p = 0; p < n_partitions; ++p) {
        // TODO: Set the active partitions in the fake_treeinfo
        optimize_branch(ann_network, oldTrees, sumtables, pmatrix_index, p, brlenOptMethod, max_iters);
    }

    // restore the network root
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        invalidatePmatrixIndex(ann_network, pmatrix_index);
    }

    double final_logl = computeLoglikelihood(ann_network);
    //std::cout << "old_logl: " << old_logl << "\n";
    //std::cout << "final_logl: " << final_logl << "\n";
    assert((final_logl >= old_logl) || (fabs(final_logl - old_logl) < 1E-3));

    return final_logl;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius,
        std::unordered_set<size_t> candidates) {
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

    Node* old_virtual_root = ann_network.network.root;

    while (!candidates.empty()) {
        size_t pmatrix_index = *candidates.begin();
        candidates.erase(candidates.begin());
        //std::cout << "optimizing branch " << pmatrix_index << "\n";

        if (act_iters[pmatrix_index] >= max_iters) {
            continue;
        }
        act_iters[pmatrix_index]++;

        double new_logl = optimize_branch(ann_network, pmatrix_index, brlenOptMethod, max_iters);

        if (new_logl - old_logl > lh_epsilon) { // add all neighbors of the branch to the candidates
            add_neighbors_in_radius(ann_network, candidates, pmatrix_index, 1);
        }
        old_logl = new_logl;
        //old_virtual_root = new_virtual_root;
    }
    if ((old_logl < start_logl) && (fabs(old_logl - start_logl) >= 1E-3)) {
        std::cout << "old_logl: " << old_logl << "\n";
        std::cout << "start_logl: " << start_logl << "\n";
        throw std::runtime_error("Overall loglikelihood got worse");
    }
    assert((old_logl >= start_logl) || (fabs(old_logl - start_logl) < 1E-3));
    
    return old_logl;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius) {
    std::unordered_set<size_t> candidates;
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        candidates.emplace(i);
    }
    return optimize_branches(ann_network, max_iters, radius, candidates);
}

/**
 * Re-infers the branch lengths of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeBranches(AnnotatedNetwork &ann_network) {
    double old_score = scoreNetwork(ann_network);

    int brlen_smooth_factor = 100;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = PLLMOD_OPT_BRLEN_OPTIMIZE_ALL;
    optimize_branches(ann_network, max_iters, radius);

    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after branch length optimization: " << new_score << "\n";

    if (new_score > old_score) {
        std::cout << "old score: " << old_score << "\n";
        std::cout << "new score: " << new_score << "\n";
        throw std::runtime_error("Complete brlenopt made BIC worse");
    }

    assert(new_score <= old_score + ann_network.options.score_epsilon);
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED && ann_network.fake_treeinfo->partition_count > 1) {
        old_score = scoreNetwork(ann_network);
        pllmod_algo_opt_brlen_scalers_treeinfo(ann_network.fake_treeinfo,
                                                        RAXML_BRLEN_SCALER_MIN,
                                                        RAXML_BRLEN_SCALER_MAX,
                                                        ann_network.options.brlen_min,
                                                        ann_network.options.brlen_max,
                                                        RAXML_PARAM_EPSILON);
        new_score = scoreNetwork(ann_network);
        std::cout << "BIC score after branch length scaler optimization: " << new_score << "\n";
        assert(new_score <= old_score + ann_network.options.score_epsilon);
    }
}

}
