/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"

#include <cassert>
#include <cmath>

#include "../Fake.hpp"

namespace netrax {

static char * xstrdup(const char * s) {
	size_t len = strlen(s);
	char * p = (char *) malloc(len + 1);
	if (!p) {
		pll_errno = PLL_ERROR_MEM_ALLOC;
		snprintf(pll_errmsg, 200, "Memory allocation failed");
		return NULL;
	}
	return strcpy(p, s);
}

void make_connections(Node* networkNode, Node* networkParentNode, pll_unode_t* unode) {
	assert(networkNode->getType() == NodeType::BASIC_NODE);
	unode->next = NULL;

	std::vector<Node*> children = networkNode->getActiveChildren(networkParentNode);
	double length_to_add = 0;
	while (children.size() == 1) { // this is the case if one of the children is a reticulation node but it's not active
		// in this case, we need to skip this node and directly connect to the next
		length_to_add += children[0]->getLink()->edge->getLength();
		children = children[0]->getActiveChildren(networkNode);
	}
	// now we should have either zero children (leaf node), or 2 children (inner tree node)
	assert(children.empty() || children.size() == 2);
	if (!children.empty()) {
		pll_unode_t* fromChild1 = pllmod_utree_create_node(children[0]->getClvIndex(), children[0]->getScalerIndex(),
				xstrdup(children[0]->getLabel().c_str()), NULL);
		fromChild1->node_index = children[0]->getLink()->node_index;
		fromChild1->length = children[0]->getLink()->edge->getLength() + length_to_add;
		fromChild1->back = unode;

		pll_unode_t* toChild1 = pllmod_utree_create_node(children[0]->getClvIndex(), children[0]->getScalerIndex(),
				xstrdup(children[0]->getLabel().c_str()), NULL);
		toChild1->node_index = children[0]->getLink()->outer->node_index;
		toChild1->length = children[0]->getLink()->edge->getLength() + length_to_add;
		toChild1->back = fromChild1;

		pll_unode_t* fromChild2 = pllmod_utree_create_node(children[1]->getClvIndex(), children[1]->getScalerIndex(),
				xstrdup(children[1]->getLabel().c_str()), NULL);
		fromChild2->node_index = children[1]->getLink()->node_index;
		fromChild2->length = children[1]->getLink()->edge->getLength() + length_to_add;
		fromChild2->back = unode;

		pll_unode_t* toChild2 = pllmod_utree_create_node(children[0]->getClvIndex(), children[0]->getScalerIndex(),
				xstrdup(children[0]->getLabel().c_str()), NULL);
		toChild2->node_index = children[1]->getLink()->outer->node_index;
		toChild2->length = children[1]->getLink()->edge->getLength() + length_to_add;
		toChild2->back = fromChild2;

		unode->next = toChild1;
		unode->next->next = toChild2;
		unode->next->next->next = unode;
		make_connections(children[0], networkNode, fromChild1);
		make_connections(children[1], networkNode, fromChild2);
	}
}

pll_utree_t * displayed_tree_to_utree(Network& network, size_t tree_index) {
	network.setReticulationParents(tree_index);

	Node* root = network.root;
	Node* root_back = network.root->getLink()->getTargetNode();

	pll_unode_t* uroot = pllmod_utree_create_node(root->getClvIndex(), root->getScalerIndex(), xstrdup(root->getLabel().c_str()), NULL);
	uroot->node_index = root->getLink()->node_index;

	if (root_back->getType() == NodeType::RETICULATION_NODE) {
		if (root_back->getReticulationData()->getLinkToActiveParent()->getTargetNode() == root) {
			// TODO
		} else {
			// TODO
		}
	}

	pll_unode_t* uroot_back = pllmod_utree_create_node(root_back->getClvIndex(), root_back->getScalerIndex(),
			xstrdup(root_back->getLabel().c_str()), NULL);
	uroot_back->node_index = root_back->getLink()->node_index;

	// TODO: Set the lengths of uroot and uroot_back... The following is only correct if neither root nor root_back is a reticulation node.
	uroot->length = root->getLink()->edge->getLength();
	uroot_back->length = root_back->getLink()->edge->getLength();

	// TODO: If root_back is a reticulation node, then we have two cases: Either we have to do nothing (if root is not the active parent of root_back), or we have to do the same reticulation node handling as in make_connections.

	uroot->back = uroot_back;
	uroot_back->back = uroot;

	make_connections(root, root_back, uroot);
	make_connections(root_back, root, uroot_back);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

// TODO: Use a dirty flag to only update CLVs that are needed... actually, we might already have it. It's called clv_valid and is in the treeinfo...

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
	operation.parent_clv_index = actNode->getClvIndex();
	operation.parent_scaler_index = actNode->getScalerIndex();
	operation.child1_clv_index = activeChildren[0]->getClvIndex();
	operation.child1_scaler_index = activeChildren[0]->getScalerIndex();
	operation.child1_matrix_index = activeChildren[0]->getEdgeTo(actNode)->getPMatrixIndex();
	if (activeChildren.size() == 2) {
		operation.child2_clv_index = activeChildren[1]->getClvIndex();
		operation.child2_scaler_index = activeChildren[1]->getScalerIndex();
		operation.child2_matrix_index = activeChildren[0]->getEdgeTo(actNode)->getPMatrixIndex();

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

	// How to do the operations at the top-level root trifurcation?
	// First with root->back, then with root...
	createOperationsPostorder(network.root, network.root->getLink()->getTargetNode(), ops, fake_clv_index, fake_pmatrix_index);
	createOperationsPostorder(network.root->getLink()->getTargetNode(), network.root, ops, fake_clv_index, fake_pmatrix_index);

	return ops;
}

// TODO: Add bool incremental...
// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Get rid of the exponentiation, as discussed in the notes when Céline was there (using the per-site-likelihoods)
double computeLoglikelihood(Network& network, const pllmod_treeinfo_t& fake_treeinfo, int incremental, int update_pmatrices) {
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
			tree_logl += pll_compute_edge_loglikelihood(fake_treeinfo.partitions[j], network.root->getClvIndex(),
					network.root->getScalerIndex(), rootBack->getClvIndex(), rootBack->getScalerIndex(),
					network.root->getLink()->edge->getPMatrixIndex(), fake_treeinfo.param_indices[j], nullptr);
			assert(tree_logl != -std::numeric_limits<double>::infinity());
		}
		network_l *= exp(tree_logl);
	}

	return log(network_l);
}

}
