/*
 * Operations.hpp
 *
 *  Created on: Jun 4, 2020
 *      Author: sarah
 */

#pragma once

#include <stddef.h>
#include <vector>

#include "../graph/Node.hpp"

namespace netrax {
struct AnnotatedNetwork;

void fill_untouched_clvs(AnnotatedNetwork &ann_network, std::vector<bool> &clv_touched,
        const std::vector<bool> &dead_nodes, size_t partition_idx, Node *startNode);

pll_operation_t buildOperationInternal(Network &network, Node *parent, Node *child1, Node *child2,
        size_t fake_clv_index, size_t fake_pmatrix_index);

std::vector<pll_operation_t> createOperations(AnnotatedNetwork &ann_network, size_t partition_idx,
        const std::vector<Node*> &parent, BlobInformation &blobInfo, unsigned int megablobIdx,
        const std::vector<bool> &dead_nodes, bool incremental, Node *displayed_tree_root);

std::vector<pll_operation_t> createOperationsUpdatedReticulation(AnnotatedNetwork &ann_network,
        size_t partition_idx, const std::vector<Node*> &parent, Node *actNode,
        const std::vector<bool> &dead_nodes, bool incremental,
        Node *displayed_tree_root);

}
