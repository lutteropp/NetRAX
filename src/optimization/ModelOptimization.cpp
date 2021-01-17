#include "ModelOptimization.hpp"

#include "../graph/AnnotatedNetwork.hpp"
#include "../likelihood/LikelihoodComputation.hpp"

namespace netrax {

/**
 * Re-infers the likelihood model parameters of a given network.
 * 
 * @param ann_network The network.
 */
void optimizeModel(AnnotatedNetwork &ann_network) {
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double old_score = scoreNetwork(ann_network);
    std::cout << "BIC score before model optimization: " << old_score << "\n";
    ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
    assert(netrax::computeLoglikelihood(ann_network, 1, 1) == netrax::computeLoglikelihood(ann_network, 0, 1));
    double new_score = scoreNetwork(ann_network);
    std::cout << "BIC score after model optimization: " << new_score << "\n";
    assert(new_score <= old_score + ann_network.options.score_epsilon);
}

}