/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

double evaluateTreesPartition(AnnotatedNetwork& ann_network, size_t partition_idx, std::vector<TreeLoglData>& treeLoglData);
void processNodeImproved(AnnotatedNetwork& ann_network, int incremental, Node* node, std::vector<Node*>& children, const ReticulationConfigSet& extraRestrictions, bool append = false);

const TreeLoglData& getMatchingOldTree(AnnotatedNetwork& ann_network, const std::vector<DisplayedTreeData>& oldTrees, const ReticulationConfigSet& queryChoices);
bool reuseOldDisplayedTreesCheck(AnnotatedNetwork& ann_network, int incremental);

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental = 1, int update_pmatrices = 1);

}
