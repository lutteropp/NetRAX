/*
 * BrlenOptTest.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: sarah
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/graph/Common.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

TEST (BrlenOptTest, testTheTest) {
	ASSERT_TRUE(true);
}

TEST (BrlenOptTest, tree) {
	// initial setup
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";
	NetraxOptions treeOptions;
	treeOptions.network_file = treePath;
	treeOptions.msa_file = msaPath;
	treeOptions.use_repeats = true;
	RaxmlWrapper treeWrapper = RaxmlWrapper(treeOptions);
	treeWrapper.enableRaxmlDebugOutput();

	Tree normalTree = Tree::loadFromFile(treePath);
	TreeInfo infoRaxml = treeWrapper.createRaxmlTreeinfo(normalTree.pll_utree_copy());
	Network treeNetwork = readNetworkFromFile(treePath);
	TreeInfo infoNetwork = treeWrapper.createRaxmlTreeinfo(treeNetwork);

	// initial logl computation
	double initial_logl_raxml = infoRaxml.loglh(false);
	std::cout << "RAXML - Initial loglikelihood: " << initial_logl_raxml << "\n";
	double initial_logl_network = infoNetwork.loglh(false);
	std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";
	ASSERT_FLOAT_EQ(initial_logl_raxml, initial_logl_network);

	ASSERT_EQ(infoRaxml.pll_treeinfo().tree->edge_count + 1, infoNetwork.pll_treeinfo().tree->edge_count);

	// Compute branch id mapping to check the branch lengths
	std::vector<size_t> utreeIDToNetworkID = getDtBranchToNetworkBranchMapping(*infoRaxml.pll_treeinfo().tree,
			*((RaxmlWrapper::NetworkParams*) (infoNetwork.pll_treeinfo().likelihood_computation_params))->network, 0);

	std::cout << "utreeIDToNetworkID.size(): " << utreeIDToNetworkID.size() << "\n";

	std::cout << "RAXML - The branch lengths before brlen optimization are:\n";
	for (size_t i = 0; i < infoRaxml.pll_treeinfo().tree->edge_count; ++i) {
		std::cout << " " << std::setprecision(17) << infoRaxml.pll_treeinfo().branch_lengths[0][i] << "\n";
	}
	std::cout << "NETWORK - The branch lengths before brlen optimization are:\n";
	for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
		std::cout << " " << std::setprecision(17) << infoNetwork.pll_treeinfo().branch_lengths[0][utreeIDToNetworkID[i]]
				<< "\n";
	}

	for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
		ASSERT_FLOAT_EQ(infoRaxml.pll_treeinfo().branch_lengths[0][i],
				infoNetwork.pll_treeinfo().branch_lengths[0][utreeIDToNetworkID[i]]);
	}

	// branch length optimization
	// problem seems to lie in the return value? The optimized brlens are the same
	double brlenopt_logl_raxml = infoRaxml.optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
	std::cout << "RAXML - Loglikelihood after branch length optimization: " << brlenopt_logl_raxml << "\n";
	double brlenopt_logl_network = infoNetwork.optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
	std::cout << "NETWORK - Loglikelihood after branch length optimization: " << brlenopt_logl_network << "\n";

	std::cout << "RAXML - The optimized branch lengths are:\n";
	for (size_t i = 0; i < infoRaxml.pll_treeinfo().tree->edge_count; ++i) {
		std::cout << " " << std::setprecision(17) << infoRaxml.pll_treeinfo().branch_lengths[0][i] << "\n";
	}
	std::cout << "NETWORK - The optimized branch lengths are:\n";
	for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
		std::cout << " " << std::setprecision(17) << infoNetwork.pll_treeinfo().branch_lengths[0][i] << "\n";
	}

	for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
		ASSERT_FLOAT_EQ(infoRaxml.pll_treeinfo().branch_lengths[0][i],
				infoNetwork.pll_treeinfo().branch_lengths[0][utreeIDToNetworkID[i]]);
	}
	// TODO: Why are the branch lengths assigned to different edges in the two cases?

	double normal_logl_raxml = infoRaxml.loglh(0);
	double normal_logl_network = infoNetwork.loglh(0);
	std::cout << "RAXML - Loglikelihood when called normally: " << normal_logl_raxml << "\n";
	std::cout << "NETWORK - Loglikelihood when called normally: " << normal_logl_network << "\n";

	ASSERT_FLOAT_EQ(brlenopt_logl_network, normal_logl_network);

	ASSERT_FLOAT_EQ(brlenopt_logl_raxml, brlenopt_logl_network);
}
