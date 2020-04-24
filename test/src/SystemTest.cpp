/*
 * SystemTest.cpp
 *
 *  Created on: Oct 30, 2019
 *      Author: sarah
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"
#include "src/graph/Network.hpp"
#include "src/Api.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/graph/Common.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "../../examples/sample_networks/";

TEST (SystemTest, testTheTest) {
    ASSERT_TRUE(true);
}

TEST (SystemTest, allTreeOldRaxml) {
    // initial setup
    std::string treePath = DATA_PATH + "tree.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.nw";

    Tree normalTree = Tree::loadFromFile(treePath);
    NetraxOptions treeOptions;
    treeOptions.network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = true;
    RaxmlWrapper treeWrapper = RaxmlWrapper(treeOptions);
    //treeWrapper.enableRaxmlDebugOutput();
    TreeInfo info = treeWrapper.createRaxmlTreeinfo(normalTree.pll_utree_copy());

    // initial logl computation
    double initial_logl = info.loglh(false);
    std::cout << "Initial loglikelihood: " << initial_logl << "\n";

    // model parameter optimization
    double modelopt_logl = info.optimize_model(treeWrapper.getRaxmlOptions().lh_epsilon);
    std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

    std::cout << "The branch lengths before brlen optimization are:\n";
    for (size_t i = 0; i < info.pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << info.pll_treeinfo().branch_lengths[0][i] << "\n";
    }

    // branch length optimization
    double brlenopt_logl = info.optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
    std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";

    std::cout << "The optimized branch lengths are:\n";
    for (size_t i = 0; i < info.pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << info.pll_treeinfo().branch_lengths[0][i] << "\n";
    }

    // model parameter optimization
    double modelopt2_logl = info.optimize_model(treeWrapper.getRaxmlOptions().lh_epsilon);
    std::cout << "Loglikelihood after model optimization again: " << modelopt2_logl << "\n";
}

TEST (SystemTest, allTree) {
    // initial setup
    std::string treePath = DATA_PATH + "tree.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.nw";
    NetraxOptions treeOptions;
    treeOptions.network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = true;

    AnnotatedNetwork ann_network = build_annotated_network(treeOptions);
    pllmod_treeinfo_t* treeinfo = ann_network.fake_treeinfo;

    // initial logl computation
    double initial_logl = computeLoglikelihood(ann_network);
    std::cout << "Initial loglikelihood: " << initial_logl << "\n";

    // model parameter optimization
    double modelopt_logl = optimizeModel(ann_network);
    std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";
    // TODO: Why does the model parameter optimization fail in this case, but not in the one above?

    std::cout << "The branch lengths before brlen optimization are:\n";
    for (size_t i = 0; i < treeinfo->tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << treeinfo->branch_lengths[0][i] << "\n";
    }

    // branch length optimization
    // TODO: Why does this give us a positive number???
    double brlenopt_logl = optimizeBranches(ann_network);
    std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";

    std::cout << "The optimized branch lengths are:\n";
    for (size_t i = 0; i < treeinfo->tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << treeinfo->branch_lengths[0][i] << "\n";
    }

    // model parameter optimization
    double modelopt2_logl = optimizeModel(ann_network);
    std::cout << "Loglikelihood after model optimization again: " << modelopt2_logl << "\n";
}

TEST (SystemTest, allNetwork) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.nw";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    AnnotatedNetwork ann_network = build_annotated_network(smallOptions);

    // initial logl computation
    double initial_logl = computeLoglikelihood(ann_network);
    std::cout << "Initial loglikelihood: " << initial_logl << "\n";

    // model parameter optimization
    double modelopt_logl = optimizeModel(ann_network);
    std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

    // branch length optimization
    // TODO: Why does this give us a positive number???
    double brlenopt_logl = optimizeBranches(ann_network);
    std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";

    // model parameter optimization
    double modelopt2_logl = optimizeModel(ann_network);
    std::cout << "Loglikelihood after model optimization again: " << modelopt2_logl << "\n";
}
