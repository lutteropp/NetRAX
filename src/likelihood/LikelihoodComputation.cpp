/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../graph/BiconnectedComponents.hpp"
#include "../graph/Node.hpp"

#include <cassert>
#include <cmath>

namespace netrax {

// TODO: Use a dirty flag to only update CLVs that are needed... actually, we might already have it. It's called clv_valid and is in the treeinfo...

void printClv(const pllmod_treeinfo_t &treeinfo, size_t clv_index, size_t partition_index) {
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
		size_t fake_pmatrix_index, const std::vector<bool>& dead_nodes) {
	std::vector<Node*> activeChildren = actNode->getActiveChildren(parent);
	if (activeChildren.empty()) { // nothing to do if we are at a leaf node
		return;
	}
	assert(activeChildren.size() <= 2);
	for (size_t i = 0; i < activeChildren.size(); ++i) {
		if (!dead_nodes[activeChildren[i]->getClvIndex()]) {
			createOperationsPostorder(actNode, activeChildren[i], ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
		}
	}
	pll_operation_t operation;
	operation.parent_clv_index = actNode->getClvIndex();
	operation.parent_scaler_index = actNode->getScalerIndex();

	if (!dead_nodes[activeChildren[0]->getClvIndex()]) {
		operation.child1_clv_index = activeChildren[0]->getClvIndex();
		operation.child1_scaler_index = activeChildren[0]->getScalerIndex();
		operation.child1_matrix_index = activeChildren[0]->getEdgeTo(actNode)->pmatrix_index;
	} else {
		operation.child1_clv_index = fake_clv_index;
		operation.child1_scaler_index = -1;
		operation.child1_matrix_index = fake_pmatrix_index;
	}

	if (activeChildren.size() == 2 && !dead_nodes[activeChildren[1]->getClvIndex()]) {
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

std::vector<pll_operation_t> createOperations(Network &network, const std::vector<Node*>& parent, BlobInformation& blobInfo, unsigned int megablobIdx, size_t treeIdx) {
	std::vector<pll_operation_t> ops;
	size_t fake_clv_index = network.nodes.size();
	size_t fake_pmatrix_index = network.edges.size();
	setReticulationParents(blobInfo, megablobIdx, treeIdx);

	std::vector<bool> dead_nodes(network.nodes.size(), false);
	fill_dead_nodes_recursive(nullptr, network.root, dead_nodes);

	// Add the children of the other megablob roots as dead nodes, such that we don't continue down there...
	for (size_t i = 0; i < blobInfo.megablob_roots.size(); ++i) {
		if (i != megablobIdx) {
			Node* actMegablobRoot = blobInfo.megablob_roots[i];
			for (Node* child : actMegablobRoot->getChildren(parent[blobInfo.megablob_roots[i]->clv_index])) {
				dead_nodes[child->clv_index] = true;
			}
		}
	}

	if (blobInfo.megablob_roots[megablobIdx] == network.root) {
		// How to do the operations at the top-level root trifurcation?
		// First with root->back, then with root...
		createOperationsPostorder(network.root, network.root->getLink()->getTargetNode(), ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
		createOperationsPostorder(network.root->getLink()->getTargetNode(), network.root, ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
	} else {
		Node* megablobRoot = blobInfo.megablob_roots[megablobIdx];
		createOperationsPostorder(parent[megablobRoot->clv_index], megablobRoot, ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
	}
	return ops;
}

std::vector<pll_operation_t> createOperations(Network &network, size_t treeIdx) {
	std::vector<pll_operation_t> ops;
	size_t fake_clv_index = network.nodes.size();
	size_t fake_pmatrix_index = network.edges.size();
	setReticulationParents(network, treeIdx);

	std::vector<bool> dead_nodes(network.nodes.size(), false);
	fill_dead_nodes_recursive(nullptr, network.root, dead_nodes);

	// How to do the operations at the top-level root trifurcation?
	// First with root->back, then with root...
	createOperationsPostorder(network.root, network.root->getLink()->getTargetNode(), ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
	createOperationsPostorder(network.root->getLink()->getTargetNode(), network.root, ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
	return ops;
}

double displayed_tree_prob(Network &network, size_t tree_index, size_t partition_index) {
	setReticulationParents(network, tree_index);
	double logProb = 0;
	for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
		logProb += log(network.reticulation_nodes[i]->getReticulationData()->getActiveProb(partition_index));
	}
	return exp(logProb);
}

double compute_tree_logl(Network &network, pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx, size_t partition_idx,
		std::vector<double> *persite_logl) {
// Create pll_operations_t array for the current displayed tree
	std::vector<pll_operation_t> ops = createOperations(network, tree_idx);
	unsigned int ops_count = ops.size();

	Node *ops_root = network.getNodeByClvIndex(ops[ops.size() - 1].parent_clv_index);

// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
	pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);

	Node *rootBack = network.root->getLink()->getTargetNode();

	double tree_partition_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx], ops_root->getClvIndex(),
			ops_root->getScalerIndex(), rootBack->getClvIndex(), rootBack->getScalerIndex(), ops_root->getLink()->edge->pmatrix_index,
			fake_treeinfo.param_indices[partition_idx], persite_logl->empty() ? nullptr : persite_logl->data());
	return tree_partition_logl;
}

// TODO: Add bool incremental...
void setup_pmatrices(Network& network, pllmod_treeinfo_t &fake_treeinfo, int incremental, int update_pmatrices) {
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
}

struct BestPersiteLoglikelihoodData {
	double best_site_logl;
	// for each reticulation: in how many displayed trees has it been taken/ not taken to get to this site_logl?
	std::vector<unsigned int> first_parent_taken_for_best_cnt;

	BestPersiteLoglikelihoodData(unsigned int reticulation_count) :
			best_site_logl(-std::numeric_limits<double>::infinity()), first_parent_taken_for_best_cnt(reticulation_count, 0) {
	}
};

void updateBestPersiteLoglikelihoods(unsigned int treeIdx, unsigned int num_reticulations, unsigned int numSites,
		std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, const std::vector<double> &persite_logl) {
	for (size_t s = 0; s < numSites; ++s) {
		if (best_persite_logl_network[s].best_site_logl < persite_logl[s]) {
			std::fill(best_persite_logl_network[s].first_parent_taken_for_best_cnt.begin(),
					best_persite_logl_network[s].first_parent_taken_for_best_cnt.end(), 0);
			best_persite_logl_network[s].best_site_logl = persite_logl[s];
		}
		if (best_persite_logl_network[s].best_site_logl == persite_logl[s]) {
			for (size_t r = 0; r < num_reticulations; ++r) {
				if (treeIdx & (1 << r)) {
					best_persite_logl_network[s].first_parent_taken_for_best_cnt[r]++;
				}
			}
		}
	}
}

void update_total_taken(std::vector<unsigned int> &totalTaken, std::vector<unsigned int> &totalNotTaken, bool unlinked_mode,
		unsigned int numSites, unsigned int num_reticulations, const std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network) {
	if (unlinked_mode) {
		std::fill(totalTaken.begin(), totalTaken.end(), 0);
		std::fill(totalNotTaken.begin(), totalNotTaken.end(), 0);
	}
	size_t n_trees = 1 << num_reticulations;
	for (size_t s = 0; s < numSites; ++s) {
		for (size_t r = 0; r < num_reticulations; ++r) {
			totalTaken[r] += best_persite_logl_network[s].first_parent_taken_for_best_cnt[r];
			totalNotTaken[r] += n_trees - best_persite_logl_network[s].first_parent_taken_for_best_cnt[r];
		}
	}
}

bool update_probs(Network& network, unsigned int partitionIdx, const std::vector<unsigned int> &totalTaken,
		const std::vector<unsigned int> &totalNotTaken) {
	bool reticulationProbsHaveChanged = false;
	for (size_t r = 0; r < network.num_reticulations(); ++r) {
		double newProb = (double) totalTaken[r] / (totalTaken[r] + totalNotTaken[r]); // Percentage of sites that were maximized when taking this reticulation
		double oldProb = network.reticulation_nodes[r]->getReticulationData()->getProb(partitionIdx);
		if (newProb != oldProb) {
			reticulationProbsHaveChanged = true;
		}
		network.reticulation_nodes[r]->getReticulationData()->setProb(newProb, partitionIdx);
	}
	return reticulationProbsHaveChanged;
}

// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Add the blobs
std::vector<double> compute_persite_lh(unsigned int partitionIdx, Network &network, pllmod_treeinfo_t &fake_treeinfo, bool unlinked_mode,
		bool update_reticulation_probs, unsigned int numSites, std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network) {
	std::vector<double> persite_lh_network(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);

	// Iterate over all displayed trees
	unsigned int num_reticulations = network.num_reticulations();
	size_t n_trees = 1 << num_reticulations;
	for (size_t treeIdx = 0; treeIdx < n_trees; ++treeIdx) {
		double tree_prob = displayed_tree_prob(network, treeIdx, unlinked_mode ? 0 : partitionIdx);
		if (tree_prob == 0.0 && !update_reticulation_probs) {
			continue;
		}

		std::vector<double> persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
		compute_tree_logl(network, fake_treeinfo, treeIdx, partitionIdx, &persite_logl);
		for (size_t s = 0; s < numSites; ++s) {
			persite_lh_network[s] += exp(persite_logl[s]) * tree_prob;
		}

		if (update_reticulation_probs) {
			updateBestPersiteLoglikelihoods(treeIdx, network.num_reticulations(), numSites, best_persite_logl_network, persite_logl);
		}
	}

	return persite_lh_network;
}

// TODO: Add bool incremental...
double processPartition(unsigned int partitionIdx, Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental,
		bool update_reticulation_probs, std::vector<unsigned int> &totalTaken, std::vector<unsigned int> &totalNotTaken, bool unlinked_mode,
		bool &reticulationProbsHaveChanged) {
	unsigned int numSites = fake_treeinfo.partitions[partitionIdx]->sites;
	std::vector<BestPersiteLoglikelihoodData> best_persite_logl_network;
	if (update_reticulation_probs) {
		best_persite_logl_network = std::vector<BestPersiteLoglikelihoodData>(numSites,
				BestPersiteLoglikelihoodData(network.num_reticulations()));
		reticulationProbsHaveChanged = false;
	}

	std::vector<double> persite_lh_network = compute_persite_lh(partitionIdx, network, fake_treeinfo, unlinked_mode,
			update_reticulation_probs, numSites, best_persite_logl_network);

	double network_partition_logl = 0.0;
	for (size_t s = 0; s < numSites; ++s) {
		network_partition_logl += log(persite_lh_network[s]);
	}

	fake_treeinfo.partition_loglh[partitionIdx] = network_partition_logl;

	if (update_reticulation_probs) {
		update_total_taken(totalTaken, totalNotTaken, unlinked_mode, numSites, network.num_reticulations(), best_persite_logl_network);
		if (unlinked_mode) {
			reticulationProbsHaveChanged = update_probs(network, partitionIdx, totalTaken, totalNotTaken);
		}
	}

	return network_partition_logl;
}

double computeLoglikelihood(Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental, int update_pmatrices,
		bool update_reticulation_probs) {
	setup_pmatrices(network, fake_treeinfo, incremental, update_pmatrices);
	const int old_active_partition = fake_treeinfo.active_partition;
	fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
	bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
	std::vector<unsigned int> totalTaken(network.num_reticulations(), 0);
	std::vector<unsigned int> totalNotTaken(network.num_reticulations(), 0);
	bool reticulationProbsHaveChanged = false;

	double network_logl = 0;

	// Iterate over all partitions
	for (size_t partitionIdx = 0; partitionIdx < fake_treeinfo.partition_count; ++partitionIdx) {
		fake_treeinfo.active_partition = partitionIdx;
		double network_partition_logl = processPartition(partitionIdx, network, fake_treeinfo, incremental, update_reticulation_probs,
				totalTaken, totalNotTaken, unlinked_mode, reticulationProbsHaveChanged);
		network_logl += network_partition_logl;
	}

	if (update_reticulation_probs && !unlinked_mode) {
		reticulationProbsHaveChanged = update_probs(network, 0, totalTaken, totalNotTaken);
	}

	/* restore original active partition */
	fake_treeinfo.active_partition = old_active_partition;

	if (update_reticulation_probs && reticulationProbsHaveChanged) {
		return computeLoglikelihood(network, fake_treeinfo, incremental, false, false);
	} else {
		return network_logl;
	}
}

double computeLoglikelihoodNaiveUtree(RaxmlWrapper &wrapper, Network &network, int incremental, int update_pmatrices) {
	(void) incremental;
	(void) update_pmatrices;

	assert(wrapper.num_partitions() == 1);

	size_t n_trees = 1 << network.reticulation_nodes.size();
	double network_l = 0.0;
// Iterate over all displayed trees
	for (size_t i = 0; i < n_trees; ++i) {
		double tree_prob = displayed_tree_prob(network, i, 0);
		if (tree_prob == 0.0) {
			continue;
		}

		pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, i);

		std::cout << "displayed tree #" << i << " as NEWICK:\n";
		char *text = pll_utree_export_newick(displayed_tree->vroot, NULL);
		std::string str(text);
		std::cout << str << "\n";
		free(text);

		TreeInfo displayedTreeinfo = wrapper.createRaxmlTreeinfo(displayed_tree);

		double tree_logl = displayedTreeinfo.loglh(0);

		assert(tree_logl != -std::numeric_limits<double>::infinity());
		network_l += exp(tree_logl) * tree_prob;
	}

	return log(network_l);
}

}
