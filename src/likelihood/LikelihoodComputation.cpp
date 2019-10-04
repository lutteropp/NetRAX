/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"

#include <cassert>
#include <cmath>

namespace netrax {

static char* xstrdup(const char *s) {
	size_t len = strlen(s);
	char *p = (char*) malloc(len + 1);
	if (!p) {
		pll_errno = PLL_ERROR_MEM_ALLOC;
		snprintf(pll_errmsg, 200, "Memory allocation failed");
		return NULL;
	}
	return strcpy(p, s);
}

pll_unode_t* create_unode(Node *network_node, bool takeLabelFromThisNode) {
	std::string label = "";
	if (takeLabelFromThisNode) {
		label = network_node->getLabel();
	} else {
		label = network_node->getLink()->getTargetNode()->getLabel();
	}
	if (label != "") {
		return pllmod_utree_create_node(network_node->getClvIndex(), network_node->getScalerIndex(), xstrdup(label.c_str()), NULL);
	} else {
		return pllmod_utree_create_node(network_node->getClvIndex(), network_node->getScalerIndex(),
		NULL, NULL);
	}
}

void make_connections(Node *networkNode, pll_unode_t *unode) {
	assert(networkNode->getType() == NodeType::BASIC_NODE);
	unode->next = NULL;

	Node *networkParentNode = networkNode->getLink()->getTargetNode();

	std::vector<Node*> children = networkNode->getActiveChildren(networkParentNode);
	double length_to_add = 0;

	Node *childParentNode = networkNode;
	while (children.size() == 1) { // this is the case if one of the children is a reticulation node but it's not active
		// in this case, we need to skip the other child node and directly connect to the next
		length_to_add += children[0]->getLink()->edge->getLength();
		Node *newChildParentNode = children[0];
		children = children[0]->getActiveChildren(childParentNode);
		childParentNode = newChildParentNode;
	}

	// now we should have either zero children (leaf node), or 2 children (inner tree node)
	assert(children.empty() || children.size() == 2);
	if (!children.empty()) {
		// TODO: if an active child is a reticulation node, we need to do some special case stuff
		double child1LenToAdd = 0.0;
		while (children[0]->getType() == NodeType::RETICULATION_NODE) {
			child1LenToAdd += children[0]->getLink()->edge->getLength();
			Node *reticulationChild = children[0]->getReticulationData()->getLinkToChild()->getTargetNode();
			if (reticulationChild->getType() == NodeType::RETICULATION_NODE
					&& reticulationChild->getReticulationData()->getLinkToActiveParent()->getTargetNode() != children[0]) {
				children[0] = nullptr;
				break;
			} else {
				children[0] = reticulationChild;
			}
		}
		double child2LenToAdd = 0.0;
		while (children[1]->getType() == NodeType::RETICULATION_NODE) {
			child2LenToAdd += children[1]->getLink()->edge->getLength();
			Node *reticulationChild = children[1]->getReticulationData()->getLinkToChild()->getTargetNode();
			if (reticulationChild->getType() == NodeType::RETICULATION_NODE
					&& reticulationChild->getReticulationData()->getLinkToActiveParent()->getTargetNode() != children[1]) {
				children[1] = nullptr;
				break;
			} else {
				children[1] = reticulationChild;
			}
		}
		assert(children[0]);
		assert(children[1]);

		assert(children[0]->getType() == NodeType::BASIC_NODE);
		assert(children[1]->getType() == NodeType::BASIC_NODE);

		pll_unode_t *fromChild1 = create_unode(children[0], true);
		fromChild1->node_index = children[0]->getLink()->node_index;
		fromChild1->length = children[0]->getLink()->edge->getLength() + length_to_add + child1LenToAdd;

		pll_unode_t *toChild1 = create_unode(children[0], false);
		toChild1->node_index = children[0]->getLink()->outer->node_index;
		toChild1->length = children[0]->getLink()->edge->getLength() + length_to_add + child1LenToAdd;
		toChild1->back = fromChild1;
		fromChild1->back = toChild1;

		pll_unode_t *fromChild2 = create_unode(children[1], true);
		fromChild2->node_index = children[1]->getLink()->node_index;
		fromChild2->length = children[1]->getLink()->edge->getLength() + length_to_add + child2LenToAdd;

		pll_unode_t *toChild2 = create_unode(children[1], false);
		toChild2->node_index = children[1]->getLink()->outer->node_index;
		toChild2->length = children[1]->getLink()->edge->getLength() + length_to_add + child2LenToAdd;
		toChild2->back = fromChild2;
		fromChild2->back = toChild2;

		unode->next = toChild1;
		unode->next->next = toChild2;
		unode->next->next->next = unode;
		make_connections(children[0], fromChild1);
		make_connections(children[1], fromChild2);
	}
}

pll_utree_t* handleRootPassiveReticulation(Network &network) {
	Node *root = network.root;
	Node *root_back = network.root->getLink()->getTargetNode();
	assert(root->getType() == NodeType::BASIC_NODE);
	assert(root_back->getType() == NodeType::RETICULATION_NODE);
	assert(root_back->getReticulationData()->getLinkToActiveParent()->getTargetNode() != root);

	// root is not the active parent of root_back
	// this is a more difficult case, as also the root changes

	Node *new_root;
	Node *other_child;
	// we need to get a non-leaf active child node as the new root.
	std::vector<Node*> activeChildren = root->getActiveChildren(root_back);
	assert(activeChildren.size() == 2);
	if (activeChildren[0]->isTip()) {
		assert(!activeChildren[1]->isTip());
		new_root = activeChildren[1];
		other_child = activeChildren[0];
	} else {
		new_root = activeChildren[0];
		other_child = activeChildren[1];
	}

	pll_unode_t *uroot = create_unode(new_root, true);
	uroot->node_index = new_root->getLink()->node_index;
	pll_unode_t *uroot_back = create_unode(other_child, true);
	uroot_back->node_index = other_child->getLink()->node_index;

	double edgeLen = new_root->getLink()->edge->getLength() + other_child->getLink()->edge->getLength();
	uroot->length = edgeLen;
	uroot_back->length = edgeLen;
	uroot->back = uroot_back;
	uroot_back->back = uroot;

	make_connections(new_root, uroot);
	make_connections(other_child, uroot_back);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

pll_utree_t* handleRootActiveReticulation(Network &network) {
	Node *root = network.root;
	Node *root_back = network.root->getLink()->getTargetNode();
	assert(root->getType() == NodeType::BASIC_NODE);
	assert(root_back->getType() == NodeType::RETICULATION_NODE);
	pll_unode_t *uroot = create_unode(root, true);

	// skip the reticulation node on its way...
	double skippedLen = 0.0;
	while (root_back->getType() == NodeType::RETICULATION_NODE) {
		skippedLen += root_back->getLink()->edge->getLength();
		root_back = root_back->getReticulationData()->getLinkToChild()->getTargetNode();
	}
	double totalLen = skippedLen + root_back->getLink()->edge->getLength();
	pll_unode_t *uroot_back = create_unode(root_back, true);
	uroot_back->node_index = root_back->getLink()->node_index;
	uroot->back = uroot_back;
	uroot_back->back = uroot;
	uroot->length = totalLen;
	uroot_back->length = totalLen;
	make_connections(root, uroot);
	make_connections(root_back, uroot_back);

	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

pll_utree_t* handleRootNormal(Network &network) {
	Node *root = network.root;
	Node *root_back = network.root->getLink()->getTargetNode();

	pll_unode_t *uroot = create_unode(root, true);
	uroot->node_index = root->getLink()->node_index;
	pll_unode_t *uroot_back = create_unode(root_back, true);
	uroot_back->node_index = root_back->getLink()->node_index;
	uroot->length = root->getLink()->edge->getLength();
	uroot_back->length = root_back->getLink()->edge->getLength();
	uroot->back = uroot_back;
	uroot_back->back = uroot;
	make_connections(root, uroot);
	make_connections(root_back, uroot_back);
	pll_utree_reset_template_indices(uroot, network.num_tips());
	return pll_utree_wraptree(uroot, network.num_tips());
}

pll_utree_t* displayed_tree_to_utree(Network &network, size_t tree_index) {
	network.setReticulationParents(tree_index);
	Node *root = network.root;
	assert(!root->isTip());
	assert(root->getType() == NodeType::BASIC_NODE);
	Node *root_back = network.root->getLink()->getTargetNode();

	if (root_back->getType() == NodeType::RETICULATION_NODE) {
		if (root_back->getReticulationData()->getLinkToActiveParent()->getTargetNode() == root) {
			return handleRootActiveReticulation(network);
		} else {
			return handleRootPassiveReticulation(network);
		}
	} else { // normal case
		return handleRootNormal(network);
	}
}

// TODO: Use a dirty flag to only update CLVs that are needed... actually, we might already have it. It's called clv_valid and is in the treeinfo...

void createOperationsPostorder(Node *parent, Node *actNode, std::vector<pll_operation_t> &ops, size_t fake_clv_index,
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
		operation.child2_matrix_index = activeChildren[1]->getEdgeTo(actNode)->getPMatrixIndex();

	} else { // activeChildren.size() == 1
		operation.child2_clv_index = fake_clv_index;
		operation.child2_scaler_index = -1;
		operation.child2_matrix_index = fake_pmatrix_index;
	}

	ops.push_back(operation);
}

std::vector<pll_operation_t> createOperations(Network &network, size_t treeIdx) {
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

double displayed_tree_prob(Network &network, size_t tree_index, size_t partition_index) {
	network.setReticulationParents(tree_index);
	double prob = 1.0;
	for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
		prob *= network.reticulation_nodes[i]->getReticulationData()->getActiveProb(partition_index);
	}
	return prob;
}

void update_reticulation_probs_internal_1(size_t tree_index, size_t num_reticulations, std::vector<double>& persite_logl,
		std::vector<std::pair<double, std::vector<std::pair<unsigned int, unsigned int> > > >& best_persite_logl_network) {
	for (size_t k = 0; k < persite_logl.size(); ++k) {
		if (best_persite_logl_network[k].first >= persite_logl[k]) {
			// find the current reticulation indices
			std::vector<bool> taken_parent(num_reticulations);
			for (size_t l = 0; l < num_reticulations; ++l) {
				taken_parent[l] = tree_index & (1 << l);
			}
			if (best_persite_logl_network[k].first == persite_logl[k]) {
				for (size_t l = 0; l < num_reticulations; ++l) {
					best_persite_logl_network[k].second[l].first += taken_parent[l];
					best_persite_logl_network[k].second[l].second += !taken_parent[l];
				}
			} else {
				best_persite_logl_network[k].first = persite_logl[k];
				for (size_t l = 0; l < num_reticulations; ++l) {
					best_persite_logl_network[k].second[l].first = taken_parent[l];
					best_persite_logl_network[k].second[l].second = !taken_parent[l];
				}
			}
		}
	}
}

void update_reticulation_probs_internal_2(Network& network, bool unlinked_mode, size_t num_reticulations, size_t partition_index,
		std::vector<unsigned int>& totalTaken, std::vector<unsigned int>& totalNotTaken,
		std::vector<std::pair<double, std::vector<std::pair<unsigned int, unsigned int> > > >& best_persite_logl_network) {
	if (unlinked_mode) {
		for (size_t l = 0; l < num_reticulations; ++l) {
			totalTaken[l] = 0;
			totalNotTaken[l] = 0;
		}
	}

	for (size_t k = 0; k < best_persite_logl_network.size(); ++k) {
		for (size_t l = 0; l < num_reticulations; ++l) {
			totalTaken[l] += best_persite_logl_network[k].second[l].first;
			totalNotTaken[l] += best_persite_logl_network[k].second[l].second;
		}
	}
	if (unlinked_mode) {
		for (size_t l = 0; l < num_reticulations; ++l) {
			double newProb = (double) totalTaken[l] / (totalTaken[l] + totalNotTaken[l]);
			network.reticulation_nodes[l]->getReticulationData()->setProb(newProb, partition_index);
		}
	}
}

// TODO: Add bool incremental...
// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Get rid of the exponentiation, as discussed in the notes when Céline was there (using the per-site-likelihoods)
double computeLoglikelihood(Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental, int update_pmatrices,
		bool update_reticulation_probs) {
	size_t n_trees = 1 << network.reticulation_nodes.size();
	double network_l = 0.0;

	const int old_active_partition = fake_treeinfo.active_partition;

	/* NOTE: in unlinked brlen mode, up-to-date brlens for partition p
	 * have to be prefetched to treeinfo->branch_lengths[p] !!! */
	bool collect_brlen = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? false : true);

	fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;

	bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
	std::vector<unsigned int> totalTaken(network.num_reticulations(), 0);
	std::vector<unsigned int> totalNotTaken(network.num_reticulations(), 0);

	// Iterate over all partitions
	for (size_t j = 0; j < fake_treeinfo.partition_count; ++j) {
		if (collect_brlen) {
			for (size_t i = 0; i < network.edges.size(); ++i) {
				fake_treeinfo.branch_lengths[0][network.edges[i].pmatrix_index] = network.edges[i].length;
			}
			// don't forget the fake entry
			fake_treeinfo.branch_lengths[0][network.edges.size()] = 0.0;
			if (update_pmatrices) {
				pllmod_treeinfo_update_prob_matrices(&fake_treeinfo, !incremental);
			}
		}

		std::vector<std::pair<double, std::vector<std::pair<unsigned int, unsigned int> > > > best_persite_logl_network;
		if (update_reticulation_probs) {
			best_persite_logl_network.resize(fake_treeinfo.partitions[j]->sites);
			for (size_t k = 0; k < fake_treeinfo.partitions[j]->sites; ++k) {
				std::vector<std::pair<unsigned int, unsigned int> > vec(network.num_reticulations(), std::make_pair(0, 0));
				best_persite_logl_network[k] = std::make_pair(0.0, vec);
			}
		}

		fake_treeinfo.active_partition = j;
		for (size_t i = 0; i < n_trees; ++i) {
			double tree_logl = 0.0;
			// Create pll_operations_t array for the current displayed tree
			std::vector<pll_operation_t> ops = createOperations(network, i);
			unsigned int ops_count = ops.size();
			// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
			pll_update_partials(fake_treeinfo.partitions[j], ops.data(), ops_count);
			// Compute loglikelihood at the root of the displayed tree in pll_compute_edge_loglikelihood. This needs an array of unsigned int (exists for each partition) param_indices.
			Node *rootBack = network.root->getLink()->getTargetNode();

			std::vector<double> persite_logl(fake_treeinfo.partitions[j]->sites, 0.0);
			tree_logl += pll_compute_edge_loglikelihood(fake_treeinfo.partitions[j], network.root->getClvIndex(),
					network.root->getScalerIndex(), rootBack->getClvIndex(), rootBack->getScalerIndex(),
					network.root->getLink()->edge->getPMatrixIndex(), fake_treeinfo.param_indices[j], persite_logl.data());

			if (update_reticulation_probs) {
				update_reticulation_probs_internal_1(i, network.num_reticulations(), persite_logl, best_persite_logl_network);
			}

			assert(tree_logl != -std::numeric_limits<double>::infinity());

			network_l += exp(tree_logl) * displayed_tree_prob(network, i, unlinked_mode ? 0 : j);
		}

		if (update_reticulation_probs) {
			update_reticulation_probs_internal_2(network, unlinked_mode, network.num_reticulations(), j, totalTaken, totalNotTaken,
					best_persite_logl_network);
		}
	}

	if (update_reticulation_probs && !unlinked_mode) {
		for (size_t l = 0; l < network.num_reticulations(); ++l) {
			double newProb = (double) totalTaken[l] / (totalTaken[l] + totalNotTaken[l]);
			network.reticulation_nodes[l]->getReticulationData()->setProb(newProb);
		}
	}

	/* restore original active partition */
	fake_treeinfo.active_partition = old_active_partition;

	return log(network_l);
}

// TODO: Add bool incremental...
// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Maybe also update reticulation probs here?
double computeLoglikelihoodLessExponentiation(Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental, int update_pmatrices,
		bool update_reticulation_probs) {
	size_t n_trees = 1 << network.reticulation_nodes.size();
	double network_logl = 0;

	const int old_active_partition = fake_treeinfo.active_partition;

	/* NOTE: in unlinked brlen mode, up-to-date brlens for partition p
	 * have to be prefetched to treeinfo->branch_lengths[p] !!! */
	bool collect_brlen = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? false : true);

	fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;

	bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
	std::vector<unsigned int> totalTaken(network.num_reticulations(), 0);
	std::vector<unsigned int> totalNotTaken(network.num_reticulations(), 0);

	// Iterate over all partitions
	for (size_t j = 0; j < fake_treeinfo.partition_count; ++j) {
		fake_treeinfo.active_partition = j;

		if (collect_brlen) {
			for (size_t i = 0; i < network.edges.size(); ++i) {
				fake_treeinfo.branch_lengths[0][network.edges[i].pmatrix_index] = network.edges[i].length;
			}
			// don't forget the fake entry
			fake_treeinfo.branch_lengths[0][network.edges.size()] = 0.0;
			if (update_pmatrices) {
				pllmod_treeinfo_update_prob_matrices(&fake_treeinfo, !incremental);
			}
		}

		std::vector<double> persite_lh_network(fake_treeinfo.partitions[j]->sites, 0.0);

		std::vector<std::pair<double, std::vector<std::pair<unsigned int, unsigned int> > > > best_persite_logl_network;
		if (update_reticulation_probs) {
			best_persite_logl_network.resize(fake_treeinfo.partitions[j]->sites);
			for (size_t k = 0; k < fake_treeinfo.partitions[j]->sites; ++k) {
				std::vector<std::pair<unsigned int, unsigned int> > vec(network.num_reticulations(), std::make_pair(0, 0));
				best_persite_logl_network[k] = std::make_pair(0.0, vec);
			}
		}

		// Iterate over all displayed trees
		for (size_t i = 0; i < n_trees; ++i) {
			// Create pll_operations_t array for the current displayed tree
			std::vector<pll_operation_t> ops = createOperations(network, i);
			unsigned int ops_count = ops.size();

			// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
			pll_update_partials(fake_treeinfo.partitions[j], ops.data(), ops_count);
			// Compute loglikelihood at the root of the displayed tree in pll_compute_edge_loglikelihood. This needs an array of unsigned int (exists for each partition) param_indices.
			Node *rootBack = network.root->getLink()->getTargetNode();

			// sites array
			std::vector<double> persite_logl(fake_treeinfo.partitions[j]->sites, 0.0);
			pll_compute_edge_loglikelihood(fake_treeinfo.partitions[j], network.root->getClvIndex(), network.root->getScalerIndex(),
					rootBack->getClvIndex(), rootBack->getScalerIndex(), network.root->getLink()->edge->getPMatrixIndex(),
					fake_treeinfo.param_indices[j], persite_logl.data());
			double tree_prob = displayed_tree_prob(network, i, unlinked_mode ? 0 : j);

			for (size_t k = 0; k < persite_logl.size(); ++k) {
				persite_lh_network[k] += exp(persite_logl[k]) * tree_prob;
			}

			if (update_reticulation_probs) {
				update_reticulation_probs_internal_1(i, network.num_reticulations(), persite_logl, best_persite_logl_network);
			}
		}

		for (size_t k = 0; k < persite_lh_network.size(); ++k) {
			network_logl += log(persite_lh_network[k]);
		}

		if (update_reticulation_probs) {
			update_reticulation_probs_internal_2(network, unlinked_mode, network.num_reticulations(), j, totalTaken, totalNotTaken,
								best_persite_logl_network);
		}
	}

	if (update_reticulation_probs && !unlinked_mode) {
		for (size_t l = 0; l < network.num_reticulations(); ++l) {
			double newProb = (double) totalTaken[l] / (totalTaken[l] + totalNotTaken[l]);
			network.reticulation_nodes[l]->getReticulationData()->setProb(newProb);
		}
	}

	/* restore original active partition */
	fake_treeinfo.active_partition = old_active_partition;

	return network_logl;
}

double computeLoglikelihoodNaiveUtree(RaxmlWrapper &wrapper, Network &network, int incremental, int update_pmatrices) {
	(void) incremental;
	(void) update_pmatrices;

	assert(wrapper.num_partitions() == 1);

	size_t n_trees = 1 << network.reticulation_nodes.size();
	double network_l = 0.0;
	// Iterate over all displayed trees
	for (size_t i = 0; i < n_trees; ++i) {
		pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, i);
		TreeInfo displayedTreeinfo = wrapper.createRaxmlTreeinfo(displayed_tree);

		double tree_logl = displayedTreeinfo.loglh(0);
		assert(tree_logl != -std::numeric_limits<double>::infinity());
		network_l += exp(tree_logl) * displayed_tree_prob(network, i, 0);
	}

	return log(network_l);
}

}
