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

const TreeLoglData& getMatchingOldTree(AnnotatedNetwork& ann_network, const std::vector<DisplayedTreeData>& oldTrees, const ReticulationConfigSet& queryChoices);

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental = 1, int update_pmatrices = 1);

}
