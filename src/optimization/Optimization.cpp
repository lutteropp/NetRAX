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
void optimizeBranches(AnnotatedNetwork &ann_network, double brlen_smooth_factor, bool silent, bool restricted_total_iters) {
    double old_score = scoreNetwork(ann_network);

    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = PLLMOD_OPT_BRLEN_OPTIMIZE_ALL;
    optimize_branches(ann_network, max_iters, max_iters, radius, restricted_total_iters);

    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::master()) std::cout << "BIC score after branch length optimization: " << new_score << "\n";

    if (new_score > old_score) {
        std::cout << "old score: " << old_score << "\n";
        std::cout << "new score: " << new_score << "\n";
        throw std::runtime_error("Complete brlenopt made BIC worse");
    }

    assert(new_score <= old_score);

    optimize_scalers(ann_network, silent);
}

void optimizeBranchesCandidates(AnnotatedNetwork &ann_network, std::unordered_set<size_t> brlenopt_candidates, double brlen_smooth_factor, bool silent, bool restricted_total_iters) {
    double old_score = scoreNetwork(ann_network);

    int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
    int radius = PLLMOD_OPT_BRLEN_OPTIMIZE_ALL;
    optimize_branches(ann_network, max_iters, max_iters, radius, brlenopt_candidates, restricted_total_iters);

    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::master()) std::cout << "BIC score after branch length optimization: " << new_score << "\n";

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
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    optimize_params(ann_network);

    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::master()) std::cout << "BIC score after model optimization: " << new_score << "\n";
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
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    optimize_reticulations(ann_network, 10);
    double new_score = scoreNetwork(ann_network);
    if (!silent && ParallelContext::master()) std::cout << "BIC score after updating reticulation probs: " << new_score << "\n";
    if (new_score > old_score) {
        throw std::runtime_error("BIC got worse after optimizing reticulation probs");
    }
    assert(new_score <= old_score);
}

bool logl_stays_same(AnnotatedNetwork& ann_network) {
    return true; // because this assertion takes too long

    double incremental = computeLoglikelihood(ann_network, 1, 1);
    double normal = computeLoglikelihood(ann_network, 0, 1);
    return (incremental == normal);
}

void optimizeAllNonTopology(AnnotatedNetwork &ann_network, OptimizeAllNonTopologyType type, bool silent) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (type == OptimizeAllNonTopologyType::QUICK) {
            std::cout << "optimizing model, reticulation probs, and branch lengths (quick mode)...\n";
        } else if (type == OptimizeAllNonTopologyType::NORMAL) {
            std::cout << "optimizing model, reticulation probs, and branch lengths (normal mode)...\n";
        } else {
            std::cout << "optimizing model, reticulation probs, and branch lengths (slow mode)...\n";
        }
    }

    silent = false;

    bool gotBetterSlow = true;

    while (gotBetterSlow) {
        gotBetterSlow = false;

        bool doBrlenOpt = true;
        bool doReticulationOpt = true;
        bool doModelOpt = true;
        double score_epsilon = 1E-3;

        assert(logl_stays_same(ann_network));
        bool gotBetter = true;
        while (gotBetter) {
            gotBetter = false;
            double score_before = scoreNetwork(ann_network);
            //assert(logl_stays_same(ann_network));

            if (doModelOpt) {
                double score_before_model = scoreNetwork(ann_network);
                //assert(logl_stays_same(ann_network));
                //assert(logl_stays_same(ann_network));
                optimizeModel(ann_network, silent);
                double score_after_model = scoreNetwork(ann_network);
                double model_improv = score_before_model - score_after_model;
                if (model_improv < score_epsilon) {
                    doModelOpt = false;
                }
            }

            if (doReticulationOpt) {
                double score_before_probs = scoreNetwork(ann_network);
                optimizeReticulationProbs(ann_network, silent);
                double score_after_probs = scoreNetwork(ann_network);
                double prob_improv = score_before_probs - score_after_probs;
                if (prob_improv < score_epsilon) {
                    doReticulationOpt = false;
                }
            }

            if (doBrlenOpt) {
                //assert(logl_stays_same(ann_network));
                double score_before_branches = scoreNetwork(ann_network);
                optimizeBranches(ann_network, 1.0, silent);
                double score_after_branches = scoreNetwork(ann_network);
                double brlen_improv = score_before_branches - score_after_branches;
                if (brlen_improv < score_epsilon) {
                    doBrlenOpt = false;
                }
            }

            double score_after = scoreNetwork(ann_network);
            assert(logl_stays_same(ann_network));

            if (score_after < score_before && type != OptimizeAllNonTopologyType::QUICK && (doBrlenOpt || doReticulationOpt || doModelOpt)) {
                gotBetter = true;
                if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                    if (!silent) std::cout << "improved bic: " << score_after << "\n";
                }
            }

            if (score_after < score_before && type == OptimizeAllNonTopologyType::SLOW) {
                gotBetterSlow = true;
            }
        }

    }
    assert(logl_stays_same(ann_network));
}

}