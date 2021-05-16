#pragma once

#include "../graph/DisplayedTreeData.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

void updateCLVsVirtualRerootTrees(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back);
double computeLoglikelihoodBrlenOpt(AnnotatedNetwork &ann_network, const std::vector<DisplayedTreeData>& oldTrees, unsigned int pmatrix_index, int incremental = 1, int update_pmatrices = 1, bool print_extra_debug_info = false);

}