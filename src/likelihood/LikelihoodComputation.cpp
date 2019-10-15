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

// TODO: Use a dirty flag to only update CLVs that are needed... actually, we might already have it. It's called clv_valid and is in the treeinfo...

void printClv(const pllmod_treeinfo_t& treeinfo, size_t clv_index, size_t partition_index) {
	size_t sites = treeinfo.partitions[partition_index]->sites;
	size_t rate_cats = treeinfo.partitions[partition_index]->rate_cats;
	size_t states = treeinfo.partitions[partition_index]->states;
	size_t states_padded = treeinfo.partitions[partition_index]->states_padded;
	std::cout << "Clv for clv_index " << clv_index << ": \n";
	for (unsigned int n = 0; n < sites; ++n) {
		for (unsigned int i = 0; i < rate_cats; ++i) {
			for (unsigned int j = 0; j < states; ++j) {
				std::cout << treeinfo.partitions[partition_index]->clv[clv_index][j + i * states_padded] << "\n";
			}
		}
	}
}

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
	operation.child1_matrix_index = activeChildren[0]->getEdgeTo(actNode)->pmatrix_index;
	if (activeChildren.size() == 2) {
		operation.child2_clv_index = activeChildren[1]->getClvIndex();
		operation.child2_scaler_index = activeChildren[1]->getScalerIndex();
		operation.child2_matrix_index = activeChildren[1]->getEdgeTo(actNode)->pmatrix_index;

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
	setReticulationParents(network, treeIdx);

	// How to do the operations at the top-level root trifurcation?
	// First with root->back, then with root...
	createOperationsPostorder(network.root, network.root->getLink()->getTargetNode(), ops, fake_clv_index, fake_pmatrix_index);
	createOperationsPostorder(network.root->getLink()->getTargetNode(), network.root, ops, fake_clv_index, fake_pmatrix_index);

	// Remove the operations at the end which always have the fake node as one of their children
	size_t num_ops_to_discard = 0;
	for (int i = ops.size() - 1; i >= 0; --i) {
		if (ops[i].child1_clv_index == fake_clv_index || ops[i].child2_clv_index == fake_clv_index) {
			num_ops_to_discard++;
		} else {
			break;
		}
	}
	for (size_t i = 0; i < num_ops_to_discard; ++i) {
		ops.pop_back();
	}

	return ops;
}

double displayed_tree_prob(Network &network, size_t tree_index, size_t partition_index) {
	setReticulationParents(network, tree_index);
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

double compute_tree_logl(Network& network, pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx, size_t partition_idx,
		std::vector<double>* persite_logl) {
// Create pll_operations_t array for the current displayed tree
	std::vector<pll_operation_t> ops = createOperations(network, tree_idx);
	unsigned int ops_count = ops.size();

	std::cout << "tree_idx: " << tree_idx << "\n";
	std::cout << "The ops are: (parent_clv_idx, child1_clv_idx, child2_clv_idx) \n";
	for (size_t i = 0; i < ops.size(); ++i) {
		std::cout << "(" << ops[i].parent_clv_index << ", " << ops[i].child1_clv_index << ", " << ops[i].child2_clv_index << ")\n";
	}

	Node *ops_root = network.getNodeByClvIndex(ops[ops.size() - 1].parent_clv_index);
	// TODO: Which one will be rootBack???

// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
	pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);
// Compute loglikelihood at the root of the displayed tree in pll_compute_edge_loglikelihood. This needs an array of unsigned int (exists for each partition) param_indices.
	Node *rootBack = network.root->getLink()->getTargetNode();

	std::cout << "rootBack has label: " << rootBack->label << "\n";

	double tree_partition_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx], ops_root->getClvIndex(),
			ops_root->getScalerIndex(), rootBack->getClvIndex(), rootBack->getScalerIndex(), ops_root->getLink()->edge->pmatrix_index,
			fake_treeinfo.param_indices[partition_idx], persite_logl->empty() ? nullptr : persite_logl->data());
	return tree_partition_logl;
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

	fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;

	bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
	std::vector<unsigned int> totalTaken(network.num_reticulations(), 0);
	std::vector<unsigned int> totalNotTaken(network.num_reticulations(), 0);

	std::cout << "The Sarah tip clv indices are: \n";
	for (size_t j = 0; j < network.tip_nodes.size(); ++j) {
		std::cout << network.tip_nodes[j]->label << " has clv_index " << network.tip_nodes[j]->clv_index << "\n";
	}

// Iterate over all partitions
	for (size_t j = 0; j < fake_treeinfo.partition_count; ++j) {
		fake_treeinfo.active_partition = j;

		std::vector<std::pair<double, std::vector<std::pair<unsigned int, unsigned int> > > > best_persite_logl_network;
		if (update_reticulation_probs) {
			best_persite_logl_network.resize(fake_treeinfo.partitions[j]->sites);
			for (size_t k = 0; k < fake_treeinfo.partitions[j]->sites; ++k) {
				std::vector<std::pair<unsigned int, unsigned int> > vec(network.num_reticulations(), std::make_pair(0, 0));
				best_persite_logl_network[k] = std::make_pair(0.0, vec);
			}
		}

		for (size_t i = 0; i < n_trees; ++i) {
			std::vector<double> persite_logl(fake_treeinfo.partitions[j]->sites, 0.0);
			double tree_partition_logl = compute_tree_logl(network, fake_treeinfo, i, j, &persite_logl);
			double tree_prob = displayed_tree_prob(network, i, unlinked_mode ? 0 : j);

			std::cout << "Clv of Q Sarah displayed tree #" << i << ":\n";
			printClv(fake_treeinfo, network.getNodeByLabel("Q")->clv_index, 0);

			if (update_reticulation_probs) {
				update_reticulation_probs_internal_1(i, network.num_reticulations(), persite_logl, best_persite_logl_network);
			}

			assert(tree_partition_logl != -std::numeric_limits<double>::infinity());

			std::cout << "sarah tree logl " << i << ": " << tree_partition_logl << "\n";

			network_l += exp(tree_partition_logl) * tree_prob;
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

	fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;

	bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
	std::vector<unsigned int> totalTaken(network.num_reticulations(), 0);
	std::vector<unsigned int> totalNotTaken(network.num_reticulations(), 0);

// Iterate over all partitions
	for (size_t j = 0; j < fake_treeinfo.partition_count; ++j) {
		fake_treeinfo.active_partition = j;

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
			std::vector<double> persite_logl(fake_treeinfo.partitions[j]->sites, 0.0);
			double tree_partition_logl = compute_tree_logl(network, fake_treeinfo, i, j, &persite_logl);

			std::cout << "norep tree logl " << i << ": " << tree_partition_logl << "\n";

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

		std::cout << "displayed tree #" << i << " as NEWICK:\n";
		char* text = pll_utree_export_newick(displayed_tree->vroot, NULL);
		std::string str(text);
		std::cout << str << "\n";
		free(text);

		TreeInfo displayedTreeinfo = wrapper.createRaxmlTreeinfo(displayed_tree);

		std::cout << "The naive tip clv indices are: \n";
		for (size_t j = 0; j < displayed_tree->tip_count + displayed_tree->inner_count; ++j) {
			if (displayed_tree->nodes[j]->clv_index < displayed_tree->tip_count) {
				std::cout << displayed_tree->nodes[j]->label << " has clv_index " << displayed_tree->nodes[j]->clv_index << "\n";
			}
		}

		double tree_logl = displayedTreeinfo.loglh(0);

		std::cout << "Clv of R naive displayed tree #" << i << ":\n";
		for (size_t j = 0; j < displayed_tree->tip_count + displayed_tree->inner_count; ++j) {
			if (strcmp(displayed_tree->nodes[j]->label, "R") == 0) {
				printClv(displayedTreeinfo.pll_treeinfo(), displayed_tree->nodes[j]->clv_index, 0);
			}
		}

		std::cout << "naive tree logl " << i << ": " << tree_logl << "\n";

		assert(tree_logl != -std::numeric_limits<double>::infinity());
		network_l += exp(tree_logl) * displayed_tree_prob(network, i, 0);
	}

	return log(network_l);
}

}
