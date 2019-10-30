/*
 * SystemTest.cpp
 *
 *  Created on: Oct 30, 2019
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

TEST (SystemTest, testTheTest) {
	ASSERT_TRUE(true);
}

TEST (SystemTest, allTreeOldRaxml) {
	// initial setup
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	Tree normalTree = Tree::loadFromFile(treePath);
	NetraxOptions treeOptions;
	treeOptions.network_file = treePath;
	treeOptions.msa_file = msaPath;
	treeOptions.use_repeats = true;
	RaxmlWrapper treeWrapper = RaxmlWrapper(treeOptions);
	TreeInfo info = treeWrapper.createRaxmlTreeinfo(normalTree.pll_utree_copy());

	// initial logl computation
	double initial_logl = info.loglh(false);
	std::cout << "Initial loglikelihood: " << initial_logl << "\n";

	// model parameter optimization
	double modelopt_logl = info.optimize_model(treeWrapper.getRaxmlOptions().lh_epsilon);
	std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

	// branch length optimization
	double brlenopt_logl = info.optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
	std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";
}

TEST (SystemTest, allTree) {
	// initial setup
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";
	Network treeNetwork = readNetworkFromFile(treePath);
	NetraxOptions treeOptions;
	treeOptions.network_file = treePath;
	treeOptions.msa_file = msaPath;
	treeOptions.use_repeats = true;
	RaxmlWrapper treeWrapper = RaxmlWrapper(treeOptions);
	TreeInfo info = treeWrapper.createRaxmlTreeinfo(treeNetwork);

	// initial logl computation
	double initial_logl = info.loglh(false);
	std::cout << "Initial loglikelihood: " << initial_logl << "\n";

	// model parameter optimization
	double modelopt_logl = info.optimize_model(treeWrapper.getRaxmlOptions().lh_epsilon);
	std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";
	// TODO: Why does the model parameter optimization fail in this case, but not in the one above?

	// branch length optimization
	double brlenopt_logl = info.optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
	std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";
}

TEST (SystemTest, allNetwork) {
	// initial setup
	std::string smallPath = "examples/sample_networks/small.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";
	Network smallNetwork = readNetworkFromFile(smallPath);
	NetraxOptions smallOptions;
	smallOptions.network_file = smallPath;
	smallOptions.msa_file = msaPath;
	smallOptions.use_repeats = true;
	RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
	TreeInfo info = smallWrapper.createRaxmlTreeinfo(smallNetwork);

	// initial logl computation
	double initial_logl = info.loglh(false);
	std::cout << "Initial loglikelihood: " << initial_logl << "\n";

	// model parameter optimization
	double modelopt_logl = info.optimize_model(smallWrapper.getRaxmlOptions().lh_epsilon);
	std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

	// branch length optimization
	double brlenopt_logl = info.optimize_branches(smallWrapper.getRaxmlOptions().lh_epsilon, 1);
	std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";
}
