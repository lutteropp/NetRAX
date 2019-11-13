/*
 * BranchLengthOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "BranchLengthOptimization.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>

#include "../graph/Common.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../RaxmlWrapper.hpp"
#include "../utils.hpp"

namespace netrax {

struct OptimizedBranchLength {
	size_t tree_index;
	double length;
	double tree_prob;
};

double optimize_branches(const NetraxOptions &options, Network &network, pllmod_treeinfo_t &fake_treeinfo, double min_brlen,
		double max_brlen, double lh_epsilon, int max_iters, int opt_method, int radius) {
	// for now, optimize branches on each of the displayed trees, exported as a pll_utree_t data structure.
	// Keep track of which branch in the exported pll_utree_t corresponds to which branch of the network data structure.
	// Run br-length optimization on each of the displayed trees.
	// For each branch in the network, collect the optimized branch lengths from the displayed trees.
	// Print the optimized displayed tree branch lengths (as well as displayed tree prob) for each branch in the network to a file.
	// for now, set the network brlens to a weighted average of the displayed-tree-brlens (weighted by tree probability)
	// also print the new network brlens.
	// Do some plots.

	size_t partitionIdx = 0;

	std::vector<double> old_brlens(network.num_branches());
	for (size_t i = 0; i < old_brlens.size(); ++i) {
		old_brlens[i] = network.edges[i].length;
	}

	std::vector<std::vector<OptimizedBranchLength> > opt_brlens(network.num_branches());
	size_t n_trees = 1 << network.reticulation_nodes.size();

	for (size_t tree_idx = 0; tree_idx < n_trees; tree_idx++) {
		setReticulationParents(network, tree_idx);
		pll_utree_t *displayed_utree = displayed_tree_to_utree(network, tree_idx);
		std::vector<std::vector<size_t> > dtBranchToNetworkBranch = getDtBranchToNetworkBranchMapping(*displayed_utree, network, tree_idx);

		// optimize brlens on the tree
		NetraxOptions opts;
		RaxmlWrapper wrapper(options);
		TreeInfo tInfo = wrapper.createRaxmlTreeinfo(displayed_utree, fake_treeinfo);
		Options raxmlOptions = wrapper.getRaxmlOptions();

		// TODO: Remove this again, it was only here because of the Slack discussion
		/*tInfo.optimize_model(raxmlOptions.lh_epsilon);
		std::cout << "displayed tree #" << tree_idx << " would like these model params:\n";
		const pll_partition_t* partition = tInfo.pll_treeinfo().partitions[0];
		print_model_params(*partition);
		std::cout << "\n";*/

		tInfo.optimize_branches(raxmlOptions.lh_epsilon, 0.25);
		const pllmod_treeinfo_t &pllmod_tInfo = tInfo.pll_treeinfo();

		// collect optimized brlens from the tree
		for (size_t i = 0; i < pllmod_tInfo.tree->edge_count; ++i) {
			if (dtBranchToNetworkBranch[i].size() == 1) {
				size_t networkBranchIdx = dtBranchToNetworkBranch[i][0];
				double new_brlen = pllmod_tInfo.branch_lengths[partitionIdx][i];
				double tree_prob = displayed_tree_prob(network, tree_idx, partitionIdx);
				opt_brlens[networkBranchIdx].push_back(OptimizedBranchLength { tree_idx, new_brlen, tree_prob });
			}
		}

		// print the displayed tree as NEWICK:
		/*std::cout << "displayed tree #" << tree_idx << " as NEWICK, after brlen opt on this tree:\n";
		char *text = pll_utree_export_newick(displayed_utree->vroot, NULL);
		std::string str(text);
		std::cout << str << "\n";
		free(text);*/

	}

	// set the network brlens to the weighted average of the displayed_tree brlens
	for (size_t i = 0; i < network.edges.size(); ++i) {
		size_t networkBranchIdx = network.edges[i].pmatrix_index;
		if (!opt_brlens[networkBranchIdx].empty()) {
			double treeProbSum = 0;
			for (size_t j = 0; j < opt_brlens[networkBranchIdx].size(); ++j) {
				treeProbSum += opt_brlens[networkBranchIdx][j].tree_prob;
			}
			double newLength = 0;
			for (size_t j = 0; j < opt_brlens[networkBranchIdx].size(); ++j) {
				double weight = opt_brlens[networkBranchIdx][j].tree_prob / treeProbSum;
				newLength += opt_brlens[networkBranchIdx][j].length * weight;
			}
			network.edges[i].length = newLength;
			fake_treeinfo.branch_lengths[partitionIdx][networkBranchIdx] = newLength;
		}
	}

	// for each network branch length, do the printing
	std::cout << std::setprecision(17);
	for (size_t i = 0; i < network.edges.size(); ++i) {
		std::cout << "Network branch " << i << ":\n";
		if (!opt_brlens[i].empty()) {
			std::cout << " Old brlen before optimization: " << old_brlens[i] << "\n";
			std::cout << " New brlen from weighted average: " << network.edges[i].length << "\n";
			for (size_t j = 0; j < opt_brlens[i].size(); ++j) {
				std::cout << "  Tree #" << opt_brlens[i][j].tree_index << ", prob = " << opt_brlens[i][j].tree_prob << ", opt_brlen = "
						<< opt_brlens[i][j].length << "\n";
			}
			assert(network.edges[i].length == fake_treeinfo.branch_lengths[partitionIdx][network.edges[i].pmatrix_index]);
		} else {
			std::cout << " This branch is has no exact presence in any displayed tree.\n";
		}
	}

	// just for debug: printing all network branch lengths for the partition 0
	std::cout << "End of BRLEN_OPT function - All network branch lengths for partition 0:\n";
	for (size_t i = 0; i < network.edges.size(); ++i) {
		std::cout << " pmatrix_idx = " << network.edges[i].pmatrix_index << " -> brlen = " << network.edges[i].length << "\n";
	}

	return -1 * computeLoglikelihoodLessExponentiation(network, fake_treeinfo, 0, 1, false);
}

}
