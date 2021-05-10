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

DisplayedTreeData& getMatchingDisplayedTreeAtNode(AnnotatedNetwork& ann_network, unsigned int partition_idx, unsigned int node_clv_index, const ReticulationConfigSet& queryChoices) {
    for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[node_clv_index].num_active_displayed_trees; ++i) {
        if (reticulationConfigsCompatible(queryChoices, ann_network.pernode_displayed_tree_data[node_clv_index].displayed_trees[i].treeLoglData.reticulationChoices)) {
            return ann_network.pernode_displayed_tree_data[node_clv_index].displayed_trees[i];
        }
    }
    throw std::runtime_error("No compatible displayed tree data found");
}

const TreeLoglData& getMatchingOldTree(const std::vector<DisplayedTreeData>& oldTrees, const ReticulationConfigSet& queryChoices) {
    for (size_t i = 0; i < oldTrees.size(); ++i) {
        if (reticulationConfigsCompatible(queryChoices, oldTrees[i].treeLoglData.reticulationChoices)) {
            return oldTrees[i].treeLoglData;
        }
    }
    std::cout << "query was:\n";
    printReticulationChoices(queryChoices);
    throw std::runtime_error("No compatible old tree data found");
}

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
