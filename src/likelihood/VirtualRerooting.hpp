#pragma once

#include "DisplayedTreeData.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

std::vector<DisplayedTreeData> extractOldTrees(AnnotatedNetwork& ann_network, Node* virtual_root);

void updateCLVsVirtualRerootTrees(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back);
double computeLoglikelihoodBrlenOpt(AnnotatedNetwork &ann_network, const std::vector<DisplayedTreeData>& oldTrees, unsigned int pmatrix_index, int incremental = 1, int update_pmatrices = 1);

}