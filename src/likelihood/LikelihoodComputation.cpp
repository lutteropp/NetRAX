/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"
#include "../traversal/Traversal.hpp"

#include <cmath>

namespace netrax {

// TODO: Use a dirty flag to only update CLVs that are needed... actually, we might already have it. It's called clv_valid and is in PartitionInfo...

void createOperationsPostorder(Node* parent, Node* actNode, std::vector<pll_operation_t>& ops, size_t fake_clv_index,
		size_t fake_pmatrix_index) {
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

void updateProbMatrices(Network& network, std::vector<PartitionInfo>& partitions, bool updateAll) {
	unsigned int pmatrix_count = network.edges.size();
	for (size_t i = 0; i < partitions.size(); ++i) {
		for (unsigned int j = 0; j < pmatrix_count; ++j) {
			if (partitions[i].pmatrix_valid[j] && !updateAll) {
				continue;
			}
			double p_brlen = partitions[i].branch_lengths[j];
			if (partitions[i].brlen_scaler != 1.0) {
				p_brlen *= partitions[i].brlen_scaler;
			}
			int ret = pll_update_prob_matrices(
					&partitions[i].pll_partition,
					partitions[i].param_indices.data(),
					&j,
					&p_brlen,
					1);
			if (!ret) {
				throw std::runtime_error("Updating the pmatrices failed");
			}
			partitions[i].pmatrix_valid[j] = true;
		}
	}
}

// TODO: Add bool incremental...
// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Get rid of the exponentiation, as discussed in the notes when Céline was there (using the per-site-likelihoods)
double computeLoglikelihood(Network& network, std::vector<PartitionInfo>& partitions) {
	size_t n_trees = 1 << network.reticulation_nodes.size();
	double network_l = 1.0;
	// Iterate over all displayed trees
	for (size_t i = 0; i < n_trees; ++i) {
		double tree_logl = 0.0;
		// Create pll_operations_t array for the current displayed tree
		std::vector<pll_operation_t> ops = createOperations(network, i);
		unsigned int ops_count = ops.size();
		// Iterate over all partitions
		for (size_t j = 0; j < partitions.size(); ++j) {
			// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
			pll_update_partials(&partitions[j].pll_partition, ops.data(), ops_count);
			// Compute loglikelihood at the root of the displayed tree in pll_compute_edge_loglikelihood. This needs an array of unsigned int (exists for each partition) param_indices.
			Node* rootBack = network.root->getLink()->getTargetNode();
			tree_logl += pll_compute_edge_loglikelihood(
					&partitions[j].pll_partition,
					network.root->getIndex(),
					network.root->getScalerIndex(),
					rootBack->getIndex(),
					rootBack->getScalerIndex(),
					network.root->getLink()->edge->getIndex(),
					partitions[j].param_indices.data(),
					nullptr);
		}
		network_l *= exp(tree_logl);
	}

	return log(network_l);
}

}
