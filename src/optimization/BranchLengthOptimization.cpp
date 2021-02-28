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

void checkLoglBeforeAfter(AnnotatedNetwork& ann_network) {
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
}

static double brent_target_networks(void *p, double x) {
    AnnotatedNetwork *ann_network = ((BrentBrlenParams*) p)->ann_network;
    size_t pmatrix_index = ((BrentBrlenParams*) p)->pmatrix_index;
    size_t partition_index = ((BrentBrlenParams*) p)->partition_index;
    double old_x = ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    double score;
    checkLoglBeforeAfter(*ann_network);

    Node* source = getSource(ann_network->network, ann_network->network.edges_by_index[pmatrix_index]);

    if (old_x == x) {
        score = -1 * computeLoglikelihoodSubnetwork(*ann_network, source, 1, 1);
        checkLoglBeforeAfter(*ann_network);
    } else {
        ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index] = x;
        invalidatePmatrixIndex(*ann_network, pmatrix_index);
        setup_pmatrices(*ann_network, true, true);
        assert(ann_network->fake_treeinfo->pmatrix_valid[partition_index][pmatrix_index]);
        score = -1 * computeLoglikelihoodSubnetwork(*ann_network, source, 1, 0);
        checkLoglBeforeAfter(*ann_network);
    }
    return score;
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

double optimize_branch(AnnotatedNetwork &ann_network, size_t pmatrix_index, size_t partition_index) {
    double min_brlen = ann_network.options.brlen_min;
    double max_brlen = ann_network.options.brlen_max;
    double tolerance = ann_network.options.tolerance;

    double start_logl = computeLoglikelihood(ann_network, 1, 1);
    checkLoglBeforeAfter(ann_network);
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    assert(start_logl == old_logl);

    double best_logl = start_logl;
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

    checkLoglBeforeAfter(ann_network);

    assert(new_brlen >= min_brlen && new_brlen <= max_brlen);

    //std::cout << "  score: " << score << "\n";
    //std::cout << "  old_brlen: " << old_brlen << ", new_brlen: " << new_brlen << "\n";
    best_logl = computeLoglikelihood(ann_network, 1, 1);
    //std::cout << " start logl for branch " << pmatrix_index << " with length " << old_brlen << ": " << start_logl
     //       << "\n";
    //std::cout << "   end logl for branch " << pmatrix_index << " with length " << new_brlen << ": " << best_logl
    //        << "\n";
    //std::cout << "\n";

    return best_logl;
}

double optimize_branch(AnnotatedNetwork &ann_network, size_t pmatrix_index) {
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    size_t n_partitions = 1;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    double logl = 0;
    ann_network.fake_treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    for (size_t p = 0; p < n_partitions; ++p) {
        // TODO: Set the active partitions in the fake_treeinfo
        logl = optimize_branch(ann_network, pmatrix_index, p);
    }
    logl = ann_network.raxml_treeinfo->loglh(true);
    // check whether new_logl >= old_logl
    if (logl < old_logl && fabs(logl - old_logl) >= 1E-3) {
        std::cout << "new_logl: " << logl << "\n";
        std::cout << "old_logl: " << old_logl << "\n";
    }
    assert((logl >= old_logl) || (fabs(logl - old_logl) < 1E-3));
    return logl;
}

double optimize_branches(AnnotatedNetwork &ann_network, int max_iters, int radius,
        std::unordered_set<size_t> candidates) {
    for (size_t idx : candidates) {
        assert(idx < ann_network.network.num_branches());
    }
    double lh_epsilon = ann_network.options.lh_epsilon;
    double old_logl = ann_network.raxml_treeinfo->loglh(true);
    double start_logl = old_logl;
    std::vector<size_t> act_iters(ann_network.network.num_branches(), 0);
    while (!candidates.empty()) {
        size_t pmatrix_index = *candidates.begin();
        candidates.erase(candidates.begin());
        //std::cout << "\noptimizing branch " << pmatrix_index << "\n";
        checkLoglBeforeAfter(ann_network);

        if (act_iters[pmatrix_index] >= max_iters) {
            continue;
        }
        act_iters[pmatrix_index]++;

        double new_logl = optimize_branch(ann_network, pmatrix_index);

        checkLoglBeforeAfter(ann_network);

        if (new_logl - old_logl > lh_epsilon) { // add all neighbors of the branch to the candidates
            add_neighbors_in_radius(ann_network, candidates, pmatrix_index, 1);
        }
        old_logl = new_logl;
    }
    assert(old_logl >= start_logl);
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
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    ann_network.raxml_treeinfo->optimize_branches(ann_network.options.lh_epsilon, 10);
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after branch length optimization: " << new_score << "\n";
    assert(new_score <= old_score + ann_network.options.score_epsilon);
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
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
