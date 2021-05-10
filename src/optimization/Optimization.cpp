#include "Optimization.hpp"

#include "BranchLengthOptimization.hpp"
#include "ModelOptimization.hpp"
#include "ReticulationOptimization.hpp"

#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/ComplexityScoring.hpp"

namespace netrax {

/**
 * Re-infers the branch lengths of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeBranches(AnnotatedNetwork &ann_network, bool silent, bool restricted_total_iters) {
    double old_score = scoreNetwork(ann_network);

    int brlen_smooth_factor = 100;
    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = PLLMOD_OPT_BRLEN_OPTIMIZE_ALL;
    optimize_branches(ann_network, max_iters, max_iters, radius, restricted_total_iters);

    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::local_proc_id() == 0) std::cout << "BIC score after branch length optimization: " << new_score << "\n";

    if (new_score > old_score) {
        std::cout << "old score: " << old_score << "\n";
        std::cout << "new score: " << new_score << "\n";
        throw std::runtime_error("Complete brlenopt made BIC worse");
    }

    assert(new_score <= old_score);

    optimize_scalers(ann_network, silent);
}

/**
 * Re-infers the likelihood model parameters of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeModel(AnnotatedNetwork &ann_network, bool silent) {
    //assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::local_proc_id() == 0) std::cout << "BIC score before model optimization: " << old_score << "\n";

    optimize_params(ann_network, ann_network.options.lh_epsilon);

    //assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::local_proc_id() == 0) std::cout << "BIC score after model optimization: " << new_score << "\n";
    assert(new_score <= old_score);
}

/**
 * Re-infers the reticulation probabilities of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeReticulationProbs(AnnotatedNetwork &ann_network, bool silent) {
    if (ann_network.network.num_reticulations() == 0) {
        return;
    }
    //assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    optimize_reticulations(ann_network, 100);
    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::local_proc_id() == 0) std::cout << "BIC score after updating reticulation probs: " << new_score << "\n";
    assert(new_score <= old_score);
}

bool logl_stays_same(AnnotatedNetwork& ann_network) {
    double incremental = computeLoglikelihood(ann_network, 1, 1);
    double normal = computeLoglikelihood(ann_network, 0, 1);
    return (incremental == normal);
}

void optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt, bool silent) {
    assert(logl_stays_same(ann_network));
    bool gotBetter = true;
    while (gotBetter) {
        gotBetter = false;
        double score_before = scoreNetwork(ann_network);
        assert(logl_stays_same(ann_network));
        optimizeModel(ann_network, silent);
        assert(logl_stays_same(ann_network));
        optimizeBranches(ann_network, silent);
        assert(logl_stays_same(ann_network));
        assert(logl_stays_same(ann_network));
        optimizeReticulationProbs(ann_network, silent);
        double score_after = scoreNetwork(ann_network);
        assert(logl_stays_same(ann_network));

        if (score_after < score_before && extremeOpt) {
            gotBetter = true;
        }
    }
    assert(logl_stays_same(ann_network));
}

}