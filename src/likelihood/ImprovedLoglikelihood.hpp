#pragma once

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct ReticulationConfigSet;
struct TreeLoglData;

double evaluateTreesPartition(
    AnnotatedNetwork &ann_network, size_t partition_idx,
    std::vector<TreeLoglData> &treeLoglData);
void processNodeImproved(AnnotatedNetwork &ann_network, int incremental,
                         Node *node, std::vector<Node *> &children,
                         const ReticulationConfigSet &extraRestrictions,
                         bool append = false);
double computeLoglikelihoodImproved(
    AnnotatedNetwork &ann_network, int incremental, int update_pmatrices);

}  // namespace netrax
