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
    std::vector<std::vector<TreeLoglData> >* oldTrees;
    BrlenOptMethod brlenOptMethod;
};

void invalidPmatrixIndexOnly(AnnotatedNetwork& ann_network, size_t pmatrix_index) {
    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        ann_network.fake_treeinfo->pmatrix_valid[partition_idx][pmatrix_index] = 0;
    }
    ann_network.cached_logl_valid = false;
}

static double brent_target_networks(void *p, double x) {
    AnnotatedNetwork *ann_network = ((BrentBrlenParams*) p)->ann_network;
    size_t pmatrix_index = ((BrentBrlenParams*) p)->pmatrix_index;
    size_t partition_index = ((BrentBrlenParams*) p)->partition_index;
    std::vector<std::vector<TreeLoglData>>* oldTrees = ((BrentBrlenParams*) p)->oldTrees;
    BrlenOptMethod brlenOptMethod = ((BrentBrlenParams*) p)->brlenOptMethod;

    double old_x = ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index];
    double score;
    if (old_x == x) {
        if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
            score = -1 * computeLoglikelihoodBrlenOpt(*ann_network, *oldTrees, pmatrix_index, 1, 1);
        } else {
            score = -1 * computeLoglikelihood(*ann_network);
        }
    } else {
        ann_network->fake_treeinfo->branch_lengths[partition_index][pmatrix_index] = x;

        if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
            invalidPmatrixIndexOnly(*ann_network, pmatrix_index);
            score = -1 * computeLoglikelihoodBrlenOpt(*ann_network, *oldTrees, pmatrix_index, 1, 1);
        } else {
            invalidatePmatrixIndex(*ann_network, pmatrix_index);
            score = -1 * computeLoglikelihood(*ann_network, 1, 1);
        }
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

double optimize_branch(AnnotatedNetwork &ann_network, std::vector<std::vector<TreeLoglData> >& oldTrees, size_t pmatrix_index, size_t partition_index, BrlenOptMethod brlenOptMethod) {
    ann_network.cached_logl_valid = false;

    double min_brlen = ann_network.options.brlen_min;
    double max_brlen = ann_network.options.brlen_max;
    double tolerance = ann_network.options.tolerance;

    double start_logl;
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        start_logl = computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1, 1);
    } else {
        start_logl = computeLoglikelihood(ann_network);
    }

    double best_logl = start_logl;
    BrentBrlenParams params;
    params.ann_network = &ann_network;
    params.pmatrix_index = pmatrix_index;
    params.partition_index = partition_index;
    params.oldTrees = &oldTrees;
    params.brlenOptMethod = brlenOptMethod;
    double old_brlen = ann_network.fake_treeinfo->branch_lengths[partition_index][pmatrix_index];

    assert(old_brlen >= min_brlen);
    assert(old_brlen <= max_brlen);

    // Do Brent's method to find a better branch length
    double score = 0;
    double f2x;
    double new_brlen = pllmod_opt_minimize_brent(min_brlen, old_brlen, max_brlen, tolerance, &score,
            &f2x, (void*) &params, &brent_target_networks);

    assert(new_brlen >= min_brlen && new_brlen <= max_brlen);

    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        best_logl = computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index, 1, 1);
    } else {
        best_logl = computeLoglikelihood(ann_network);
    }
    assert(best_logl >= start_logl);

    return best_logl;
}

double optimize_branch(AnnotatedNetwork &ann_network, size_t pmatrix_index, BrlenOptMethod brlenOptMethod) {
    double old_logl = computeLoglikelihood(ann_network);
    std::vector<std::vector<TreeLoglData> > oldTrees = extractOldTrees(ann_network, ann_network.network.root);

    // step 1: Do the virtual rerooting.
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        Node* old_virtual_root = ann_network.network.root;
        Node* new_virtual_root = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
        if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
            Node* new_virtual_root_back = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
            updateCLVsVirtualRerootTrees(ann_network, old_virtual_root, new_virtual_root, new_virtual_root_back);
        }
        ann_network.cached_logl_valid = false;
        assert(old_logl == computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index));
    }

    size_t n_partitions = 1;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    ann_network.fake_treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    for (size_t p = 0; p < n_partitions; ++p) {
        // TODO: Set the active partitions in the fake_treeinfo
        optimize_branch(ann_network, oldTrees, pmatrix_index, p, brlenOptMethod);
    }

    // restore the network root
    if (brlenOptMethod != BrlenOptMethod::BRENT_NORMAL) {
        invalidatePmatrixIndex(ann_network, pmatrix_index);
    }

    assert((logl >= old_logl) || (fabs(logl - old_logl) < 1E-3));

    return computeLoglikelihood(ann_network);
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

    Node* old_virtual_root = ann_network.network.root;

    while (!candidates.empty()) {
        size_t pmatrix_index = *candidates.begin();
        candidates.erase(candidates.begin());
        //std::cout << "optimizing branch " << pmatrix_index << "\n";

        if (act_iters[pmatrix_index] >= max_iters) {
            continue;
        }
        act_iters[pmatrix_index]++;

        double new_logl = optimize_branch(ann_network, pmatrix_index, brlenOptMethod);

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

    int brlen_smooth_factor = 10;
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
