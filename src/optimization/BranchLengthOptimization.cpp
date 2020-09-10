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

struct OptimizedBranchLength {
    size_t tree_index;
    double length;
    double tree_prob;
};

double computeVariance(const std::vector<OptimizedBranchLength> &brlens) {
    // TODO: Maybe adapt to different tree probabilities, such as here?
    // https://en.wikipedia.org/wiki/Variance#Discrete_random_variable
    double var = 0;
    if (brlens.empty()) {
        var = std::numeric_limits<double>::infinity();
    } else {
        // compute the mean branch length
        double mean = 0;
        for (size_t i = 0; i < brlens.size(); ++i) {
            mean += brlens[i].length;
        }
        mean /= (double) brlens.size();

        // compute the variance
        for (size_t i = 0; i < brlens.size(); ++i) {
            var += (brlens[i].length - mean) * (brlens[i].length - mean);
        }
        var /= (double) brlens.size();
    }
    return var;
}

struct BrentBrlenParams {
    AnnotatedNetwork *ann_network;
    size_t pmatrix_index;
    size_t partition_index;
};

static double brent_target_networks(void *p, double x) {
    AnnotatedNetwork *ann_network = ((BrentBrlenParams*) p)->ann_network;
    size_t pmatrix_index = ((BrentBrlenParams*) p)->pmatrix_index;
    size_t partition_index = ((BrentBrlenParams*) p)->partition_index;
    double old_x = ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    double score;
    if (old_x == x) {
        score = -1 * computeLoglikelihood(*ann_network, 1, 1, false);
    } else {
        ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index] = x;
        ann_network->network.edges_by_index[pmatrix_index]->length = x;
        ann_network->fake_treeinfo->pmatrix_valid[partition_index][pmatrix_index] = 0;
        invalidateHigherCLVs(*ann_network,
                getSource(ann_network->network, ann_network->network.edges_by_index[pmatrix_index]),
                true);
        score = -1 * computeLoglikelihood(*ann_network, 1, 1, false);
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

    double start_logl = computeLoglikelihood(ann_network, 1, 1, false);
    double best_logl = start_logl;
    if (*act_iters >= max_iters) {
        return best_logl;
    }
    BrentBrlenParams params;
    params.ann_network = &ann_network;
    params.pmatrix_index = pmatrix_index;
    params.partition_index = partition_index;
    double old_brlen = ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    // Do Brent's method to find a better branch length
    //std::cout << " optimizing branch " << pmatrix_index << ":\n";
    double score = 0;
    double f2x;
    double new_brlen = pllmod_opt_minimize_brent(min_brlen, old_brlen, max_brlen, tolerance, &score,
            &f2x, (void*) &params, &brent_target_networks);
    //std::cout << "  score: " << score << "\n";
    //std::cout << "  old_brlen: " << old_brlen << ", new_brlen: " << new_brlen << "\n";
    best_logl = computeLoglikelihood(ann_network, 1, 1, false);
    //std::cout << " start logl for branch " << pmatrix_index << " with length " << start_brlen << ": " << start_logl
    //        << "\n";
    //std::cout << "   end logl for branch " << pmatrix_index << " with length " << new_brlen << ": " << best_logl
    //        << "\n";
    //std::cout << "\n";
    (*act_iters)++;
    return best_logl;
}

double optimize_branch(AnnotatedNetwork &ann_network, int max_iters, int *act_iters,
        size_t pmatrix_index) {
    size_t n_partitions = 1;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    double logl = 0;
    for (size_t p = 0; p < n_partitions; ++p) {
        // TODO: Set the active partitions in the fake_treeinfo
        logl += optimize_branch(ann_network, max_iters, act_iters, pmatrix_index, p);
    }
    return logl;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius,
        std::unordered_set<size_t> &candidates) {
    double lh_epsilon = ann_network.options.lh_epsilon;
    int act_iters = 0;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    while (!candidates.empty()) {
        size_t pmatrix_index = *candidates.begin();
        candidates.erase(candidates.begin());
        //std::cout << "\noptimizing branch " << pmatrix_index << "\n";
        double new_logl = optimize_branch(ann_network, max_iters, &act_iters, pmatrix_index);

        //std::cout << "old logl: " << old_logl << ", new_logl: " << new_logl << "\n";

        // check whether new_logl >= old_logl
        if (new_logl < old_logl && fabs(new_logl - old_logl) >= 1E-3) {
            std::cout << "new_logl: " << new_logl << "\n";
            std::cout << "old_logl: " << old_logl << "\n";
        }
        assert(new_logl >= old_logl || fabs(new_logl - old_logl) < 1E-3);
        if (new_logl - old_logl > lh_epsilon) { // add all neighbors of the branch to the candidates
            std::unordered_set<size_t> neighbor_indices = getNeighborPmatrixIndices(
                    ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
            for (size_t idx : neighbor_indices) {
                candidates.emplace(idx);
            }
            //std::cout << "Improved brlen-opt logl: " << new_logl << "\n";
            //std::cout << "(Old logl was: " << old_logl << ")\n";
        }
        old_logl = new_logl;
    }
    return old_logl;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius) {
    std::unordered_set<size_t> candidates;
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        candidates.emplace(ann_network.network.edges[i].pmatrix_index);
    }
    return optimize_branches(ann_network, max_iters, radius, candidates);
}

}
