#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/DisplayedTreeData.hpp"

namespace netrax {

void updateCLVsVirtualRerootTrees(AnnotatedNetwork &ann_network,
                                  Node *old_virtual_root,
                                  Node *new_virtual_root,
                                  Node *new_virtual_root_back,
                                  ReticulationConfigSet &restrictions);
double computeLoglikelihoodBrlenOpt(
    AnnotatedNetwork &ann_network,
    const std::vector<DisplayedTreeData> &oldTrees, unsigned int pmatrix_index,
    int update_pmatrices = 1, bool print_extra_debug_info = false);

}  // namespace netrax