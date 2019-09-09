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

void updateProbMatrices(Network& network, pllmod_treeinfo_t& fake_treeinfo, bool updateAll) {
	unsigned int pmatrix_count = network.edges.size();
	for (size_t i = 0; i < fake_treeinfo.partition_count; ++i) {
		for (unsigned int j = 0; j < pmatrix_count; ++j) {
			if (fake_treeinfo.pmatrix_valid[i][j] && !updateAll) {
				continue;
			}
			double p_brlen = fake_treeinfo.branch_lengths[i][j];
			if (fake_treeinfo.brlen_scalers[i] != 1.0) {
				p_brlen *= fake_treeinfo.brlen_scalers[i];
			}
			int ret = pll_update_prob_matrices(
					fake_treeinfo.partitions[i],
					fake_treeinfo.param_indices[i],
					&j,
					&p_brlen,
					1);
			if (!ret) {
				throw std::runtime_error("Updating the pmatrices failed");
			}

			fake_treeinfo.pmatrix_valid[i][j] = true;
		}
	}
}

// TODO: Add bool incremental...
// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Get rid of the exponentiation, as discussed in the notes when Céline was there (using the per-site-likelihoods)
double computeLoglikelihood(Network& network, pllmod_treeinfo_t& fake_treeinfo) {
	size_t n_trees = 1 << network.reticulation_nodes.size();
	double network_l = 1.0;
	// Iterate over all displayed trees
	for (size_t i = 0; i < n_trees; ++i) {
		double tree_logl = 0.0;
		// Create pll_operations_t array for the current displayed tree
		std::vector<pll_operation_t> ops = createOperations(network, i);
		unsigned int ops_count = ops.size();
		// Iterate over all partitions
		for (size_t j = 0; j < fake_treeinfo.partition_count; ++j) {
			// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
			pll_update_partials(fake_treeinfo.partitions[j], ops.data(), ops_count);
			// Compute loglikelihood at the root of the displayed tree in pll_compute_edge_loglikelihood. This needs an array of unsigned int (exists for each partition) param_indices.
			Node* rootBack = network.root->getLink()->getTargetNode();
			tree_logl += pll_compute_edge_loglikelihood(
					fake_treeinfo.partitions[j],
					network.root->getIndex(),
					network.root->getScalerIndex(),
					rootBack->getIndex(),
					rootBack->getScalerIndex(),
					network.root->getLink()->edge->getIndex(),
					fake_treeinfo.param_indices[j],
					nullptr);
		}
		network_l *= exp(tree_logl);
	}

	return log(network_l);
}

}
