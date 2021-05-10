/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"
#include "../helper/Helper.hpp"

#include "PseudoLoglikelihood.hpp"
#include "ImprovedLoglikelihood.hpp"

namespace netrax {

void printDisplayedTreesChoices(AnnotatedNetwork& ann_network, size_t partition_idx, Node* virtualRoot) {
    NodeDisplayedTreeData& nodeData = ann_network.pernode_displayed_tree_data[virtualRoot->clv_index];
    for (size_t i = 0; i < nodeData.num_active_displayed_trees; ++i) {
        printReticulationChoices(nodeData.displayed_trees[i].treeLoglData.reticulationChoices);
    }
}

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    //just for debug
    //incremental = 0;
    //update_pmatrices = 1;
    if (ann_network.options.likelihood_variant == LikelihoodVariant::SARAH_PSEUDO) {
        return computePseudoLoglikelihood(ann_network, incremental, update_pmatrices);
    } else {
        return computeLoglikelihoodImproved(ann_network, incremental, update_pmatrices);
    }
    //return computeLoglikelihoodNaiveUtree(ann_network, incremental, update_pmatrices);
}

}
