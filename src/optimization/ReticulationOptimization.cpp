
#include "ReticulationOptimization.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>

#include "../helper/NetworkFunctions.hpp"
#include "../helper/Helper.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../RaxmlWrapper.hpp"
#include "../utils.hpp"
#include "../graph/AnnotatedNetwork.hpp"

#include "../likelihood/ComplexityScoring.hpp"

namespace netrax {

struct BrentBrprobParams {
    AnnotatedNetwork *ann_network;
    size_t reticulation_index;
};

static double brent_target_networks_prob(void *p, double x) {
    AnnotatedNetwork *ann_network = ((BrentBrprobParams*) p)->ann_network;
    size_t reticulation_index = ((BrentBrprobParams*) p)->reticulation_index;
    double old_x = ann_network->reticulation_probs[reticulation_index];
    double score;
    if (old_x == x) {
        score = -1 * computeLoglikelihood(*ann_network, 1, 1);
    } else {
        ann_network->reticulation_probs[reticulation_index] = x;
        ann_network->cached_logl_valid = false;

        if (ann_network->options.likelihood_variant == LikelihoodVariant::SARAH_PSEUDO) {
            invalidateHigherCLVs(*ann_network, ann_network->network.reticulation_nodes[reticulation_index], false);
        }

        score = -1 * computeLoglikelihood(*ann_network, 1, 1);
        //std::cout << "    score: " << score << ", x: " << x << ", old_x: " << old_x << ", pmatrix index:"
        //        << pmatrix_index << "\n";
    }
    return score;
}

double optimize_reticulation_linear_search(AnnotatedNetwork &ann_network, size_t reticulation_index) {
    double best_prob = ann_network.reticulation_probs[reticulation_index];
    double best_logl = computeLoglikelihood(ann_network);

    double step = 1.0/100;

    for (int i = 0; i <= 1/step; ++i) {
        double mid = i * step;

        ann_network.reticulation_probs[reticulation_index] = mid;
        ann_network.cached_logl_valid = false;
        if (ann_network.options.likelihood_variant == LikelihoodVariant::SARAH_PSEUDO) {
            invalidateHigherCLVs(ann_network, ann_network.network.reticulation_nodes[reticulation_index], false);
        }

        double act_logl = computeLoglikelihood(ann_network);

        if (act_logl > best_logl) {
            best_logl = act_logl;
            best_prob = mid;
        }
    }

    ann_network.reticulation_probs[reticulation_index] = best_prob;
    ann_network.cached_logl_valid = false;
    return best_prob;
}

double optimize_reticulation(AnnotatedNetwork &ann_network, size_t reticulation_index) {
    //return optimize_reticulation_linear_search(ann_network, reticulation_index);

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

    // Do Brent's method to find a better branch length
    //std::cout << " optimizing branch " << pmatrix_index << ":\n";
    double score = 0;
    double f2x;
    double new_brprob = pllmod_opt_minimize_brent(min_brprob, old_brprob, max_brprob, tolerance, &score,
            &f2x, (void*) &params, &brent_target_networks_prob);
    ann_network.reticulation_probs[reticulation_index] = new_brprob;
    ann_network.cached_logl_valid = false;
    if (ann_network.options.likelihood_variant == LikelihoodVariant::SARAH_PSEUDO) {
        invalidateHigherCLVs(ann_network, ann_network.network.reticulation_nodes[reticulation_index], false);
    }

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

}
