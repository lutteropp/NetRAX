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

#include "../graph/Common.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../RaxmlWrapper.hpp"
#include "../utils.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct BrentBrlenParams {
    AnnotatedNetwork *ann_network;
    size_t pmatrix_index;
    size_t partition_index;
};

struct BrentBrprobParams {
    AnnotatedNetwork *ann_network;
    size_t reticulation_index;
};

static double brent_target_networks(void *p, double x) {
    AnnotatedNetwork *ann_network = ((BrentBrlenParams*) p)->ann_network;
    size_t pmatrix_index = ((BrentBrlenParams*) p)->pmatrix_index;
    size_t partition_index = ((BrentBrlenParams*) p)->partition_index;
    double old_x = ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    double score;
    if (old_x == x) {
        score = -1 * computeLoglikelihood(*ann_network, 1, 1);
    } else {
        ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index] = x;
        ann_network->fake_treeinfo->pmatrix_valid[partition_index][pmatrix_index] = 0;
        setup_pmatrices(*ann_network, false, true);
        invalidateHigherCLVs(*ann_network,
                getTarget(ann_network->network, ann_network->network.edges_by_index[pmatrix_index]),
                true);
        score = -1 * computeLoglikelihood(*ann_network, 1, 1);
        assert(ann_network->fake_treeinfo->pmatrix_valid[partition_index][pmatrix_index]);
        //std::cout << "    score: " << score << ", x: " << x << ", old_x: " << old_x << ", pmatrix index:"
        //        << pmatrix_index << "\n";
    }
    return score;
}

static double brent_target_networks_prob(void *p, double x) {
    AnnotatedNetwork *ann_network = ((BrentBrprobParams*) p)->ann_network;
    size_t reticulation_index = ((BrentBrprobParams*) p)->reticulation_index;
    double old_x = ann_network->reticulation_probs[reticulation_index];
    double score;
    if (old_x == x) {
        score = -1 * computeLoglikelihood(*ann_network, 1, 1);
    } else {
        ann_network->reticulation_probs[reticulation_index] = x;

        if (!ann_network->old_displayed_trees.empty()) {
            for (size_t p = 0; p < ann_network->fake_treeinfo->partition_count; ++p) {
                for (size_t i = 0; i < ann_network->old_displayed_trees[p].size(); ++i) {
                    ann_network->old_displayed_trees[p][i].tree_logprob = displayed_tree_logprob(*ann_network, ann_network->old_displayed_trees[p][i].tree_idx);
                }
            }
        }

        score = -1 * computeLoglikelihood(*ann_network, 1, 1);
        //std::cout << "    score: " << score << ", x: " << x << ", old_x: " << old_x << ", pmatrix index:"
        //        << pmatrix_index << "\n";
    }
    return score;
}

double optimize_branch(AnnotatedNetwork &ann_network, int max_iters, int *act_iters,
        size_t pmatrix_index, size_t partition_index) {
    double min_brlen = ann_network.options.brlen_min;
    double max_brlen = ann_network.options.brlen_max;
    double tolerance = ann_network.options.tolerance;

    double start_logl = computeLoglikelihood(ann_network, 1, 1);
    double recomputed_logl = computeLoglikelihood(ann_network, 0, 1);
    assert(start_logl == recomputed_logl);
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    assert(start_logl == old_logl);

    double best_logl = start_logl;
    if (*act_iters >= max_iters) {
        return best_logl;
    }
    BrentBrlenParams params;
    params.ann_network = &ann_network;
    params.pmatrix_index = pmatrix_index;
    params.partition_index = partition_index;
    double old_brlen = ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index];

    assert(old_brlen >= min_brlen);
    assert(old_brlen <= max_brlen);

    // Do Brent's method to find a better branch length
    //std::cout << " optimizing branch " << pmatrix_index << ":\n";
    double score = 0;
    double f2x;
    double new_brlen = pllmod_opt_minimize_brent(min_brlen, old_brlen, max_brlen, tolerance, &score,
            &f2x, (void*) &params, &brent_target_networks);

    assert(new_brlen >= min_brlen && new_brlen <= max_brlen);

    //std::cout << "  score: " << score << "\n";
    //std::cout << "  old_brlen: " << old_brlen << ", new_brlen: " << new_brlen << "\n";
    best_logl = computeLoglikelihood(ann_network, 1, 1);
    //std::cout << " start logl for branch " << pmatrix_index << " with length " << old_brlen << ": " << start_logl
     //       << "\n";
    //std::cout << "   end logl for branch " << pmatrix_index << " with length " << new_brlen << ": " << best_logl
    //        << "\n";
    //std::cout << "\n";

    (*act_iters)++;
    return best_logl;
}

double optimize_reticulation(AnnotatedNetwork &ann_network, size_t reticulation_index) {
    double min_brprob = ann_network.options.brprob_min;
    double max_brprob = ann_network.options.brprob_max;
    double tolerance = ann_network.options.tolerance;

    double start_logl = computeLoglikelihood(ann_network, 1, 1);
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    assert(start_logl == old_logl);

    double best_logl = start_logl;
    BrentBrprobParams params;
    params.ann_network = &ann_network;
    params.reticulation_index = reticulation_index;
    double old_brprob = ann_network.reticulation_probs[reticulation_index];

    assert(old_brprob >= min_brprob);
    assert(old_brprob <= max_brprob);

    // Do Brent's method to find a better branch length
    //std::cout << " optimizing branch " << pmatrix_index << ":\n";
    double score = 0;
    double f2x;
    double new_brprob = pllmod_opt_minimize_brent(min_brprob, old_brprob, max_brprob, tolerance, &score,
            &f2x, (void*) &params, &brent_target_networks_prob);
    ann_network.reticulation_probs[reticulation_index] = new_brprob;

    //std::cout << "old prob for reticulation " << reticulation_index << ": " << old_brprob << "\n";
    //std::cout << "new prob for reticulation " << reticulation_index << ": " << new_brprob << "\n";

    assert(new_brprob >= min_brprob && new_brprob <= max_brprob);

    //std::cout << "  score: " << score << "\n";
    //std::cout << "  old_brlen: " << old_brlen << ", new_brlen: " << new_brlen << "\n";
    best_logl = computeLoglikelihood(ann_network, 1, 1);

    //std::cout << " start logl for branch " << pmatrix_index << " with length " << start_brlen << ": " << start_logl
    //        << "\n";
    //std::cout << "   end logl for branch " << pmatrix_index << " with length " << new_brlen << ": " << best_logl
    //        << "\n";
    //std::cout << "\n";
    return best_logl;
}

double optimize_branch(AnnotatedNetwork &ann_network, int max_iters, int *act_iters,
        size_t pmatrix_index) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    size_t n_partitions = 1;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    double logl = 0;
    for (size_t p = 0; p < n_partitions; ++p) {
        // TODO: Set the active partitions in the fake_treeinfo
        logl = optimize_branch(ann_network, max_iters, act_iters, pmatrix_index, p);
    }
    // check whether new_logl >= old_logl
    if (logl < old_logl && fabs(logl - old_logl) >= 1E-3) {
        std::cout << "new_logl: " << logl << "\n";
        std::cout << "old_logl: " << old_logl << "\n";
    }
    assert((logl >= old_logl) || (fabs(logl - old_logl) < 1E-3));
    return logl;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius,
        std::unordered_set<size_t> &candidates) {
    double lh_epsilon = ann_network.options.lh_epsilon;
    int act_iters = 0;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double start_logl = old_logl;
    while (!candidates.empty()) {
        size_t pmatrix_index = *candidates.begin();
        candidates.erase(candidates.begin());
        //std::cout << "\noptimizing branch " << pmatrix_index << "\n";
        double new_logl = optimize_branch(ann_network, max_iters, &act_iters, pmatrix_index);

        double start_logl = computeLoglikelihood(ann_network, 1, 1);
        double recomputed_logl = computeLoglikelihood(ann_network, 0, 1);
        assert(start_logl == recomputed_logl);

        if (new_logl - old_logl > lh_epsilon) { // add all neighbors of the branch to the candidates
            std::unordered_set<size_t> neighbor_indices = getNeighborPmatrixIndices(
                    ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
            for (size_t idx : neighbor_indices) {
                candidates.emplace(idx);
            }
        }
        old_logl = new_logl;
    }
    assert(old_logl >= start_logl);
    return old_logl;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius) {
    std::unordered_set<size_t> candidates;
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        candidates.emplace(ann_network.network.edges[i].pmatrix_index);
    }
    return optimize_branches(ann_network, max_iters, radius, candidates);
}

double optimize_reticulations(AnnotatedNetwork &ann_network, int max_iters) {
    double act_logl = ann_network.raxml_treeinfo->loglh(true);
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

}
