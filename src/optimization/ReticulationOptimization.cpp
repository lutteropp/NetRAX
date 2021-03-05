
#include "ReticulationOptimization.hpp"

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

        score = -1 * computeLoglikelihood(*ann_network, 1, 1);
        //std::cout << "    score: " << score << ", x: " << x << ", old_x: " << old_x << ", pmatrix index:"
        //        << pmatrix_index << "\n";
    }
    return score;
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


/**
 * Re-infers the reticulation probabilities of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeReticulationProbs(AnnotatedNetwork &ann_network) {
    if (ann_network.network.num_reticulations() == 0) {
        return;
    }
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    netrax::optimize_reticulations(ann_network, 100);
    double new_score = scoreNetwork(ann_network);
    //std::cout << "BIC score after updating reticulation probs: " << new_score << "\n";
    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

}
