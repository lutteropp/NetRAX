/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"
#include "../traversal/Traversal.hpp"

namespace netrax {

// TODO: Use a dirty flag to only update CLVs that are needed...

void createOperationsPostorder(Node* parent, Node* actNode, std::vector<pll_operation_t>& ops, size_t fake_clv_index, size_t fake_pmatrix_index) {
	std::vector<Node*> activeChildren = actNode->getActiveChildren(parent);
	if (activeChildren.empty()) { // nothing to do if we are at a leaf node
		return;
	}
	assert(activeChildren.size() <= 2);
	for (size_t i = 0; i < activeChildren.size(); ++i) {
		createOperationsPostorder(actNode, activeChildren[i], ops, fake_clv_index, fake_pmatrix_index);
	}
	pll_operation_t operation;
	operation.parent_clv_index = actNode->getIndex();
	operation.parent_scaler_index = actNode->getScalerIndex();
	operation.child1_clv_index = activeChildren[0]->getIndex();
	operation.child1_scaler_index = activeChildren[0]->getScalerIndex();
	operation.child1_matrix_index = activeChildren[0]->getEdgeTo(actNode)->getIndex();
	if (activeChildren.size() == 2) {
		operation.child2_clv_index = activeChildren[1]->getIndex();
		operation.child2_scaler_index = activeChildren[1]->getScalerIndex();
		operation.child2_matrix_index = activeChildren[0]->getEdgeTo(actNode)->getIndex();

	} else { // activeChildren.size() == 1
		operation.child2_clv_index = fake_clv_index;
		operation.child2_scaler_index = -1;
		operation.child2_matrix_index = fake_pmatrix_index;
	}

	ops.push_back(operation);
}

std::vector<pll_operation_t> createOperations(Network& network, size_t treeIdx) {
	std::vector<pll_operation_t> ops;
	size_t fake_clv_index = network.nodes.size();
	size_t fake_pmatrix_index = network.edges.size();

	network.setReticulationParents(treeIdx);
	createOperationsPostorder(nullptr, network.root, ops, fake_clv_index, fake_pmatrix_index);
	return ops;
}

void updateProbMatrices(Network& network, PartitionInfo& partition) {
	throw std::runtime_error("Not implemented yet");
}

double computeLoglikelihood(Network& network, std::vector<PartitionInfo>& partitions) {

	// Iterate over all displayed trees

		// Compute pll_operations_t array

		// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.

		// Compute loglikelihood at the root of the displaxed tree in pll_compute_edge_loglikelihood. This needs an array of unsigned int (exists for each partition) param_indices.

	return 0;
}

}
