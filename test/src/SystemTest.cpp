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

const std::string DATA_PATH = "examples/sample_networks/";

TEST (SystemTest, testTheTest) {
    ASSERT_TRUE(true);
}

TEST (SystemTest, allTreeOldRaxml) {
    // initial setup
    std::string treePath = DATA_PATH + "tree.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";

    Tree normalTree = Tree::loadFromFile(treePath);
    NetraxOptions treeOptions;
    treeOptions.network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = true;
    RaxmlWrapper treeWrapper = RaxmlWrapper(treeOptions);
    //treeWrapper.enableRaxmlDebugOutput();
    TreeInfo *info = treeWrapper.createRaxmlTreeinfo(normalTree.pll_utree_copy());

    // initial logl computation
    double initial_logl = info->loglh(false);
    std::cout << "Initial loglikelihood: " << initial_logl << "\n";

    // model parameter optimization
    double modelopt_logl = info->optimize_model(treeWrapper.getRaxmlOptions().lh_epsilon);
    std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

    std::cout << "The branch lengths before brlen optimization are:\n";
    for (size_t i = 0; i < info->pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << info->pll_treeinfo().branch_lengths[0][i] << "\n";
    }

    // branch length optimization
    double brlenopt_logl = info->optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
    std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";

    std::cout << "The optimized branch lengths are:\n";
    for (size_t i = 0; i < info->pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << info->pll_treeinfo().branch_lengths[0][i] << "\n";
    }

    // model parameter optimization
    double modelopt2_logl = info->optimize_model(treeWrapper.getRaxmlOptions().lh_epsilon);
    std::cout << "Loglikelihood after model optimization again: " << modelopt2_logl << "\n";

    delete info;
}

TEST (SystemTest, allTree) {
    // initial setup
    std::string treePath = DATA_PATH + "tree.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions treeOptions;
    treeOptions.network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = true;

    AnnotatedNetwork ann_network = build_annotated_network(treeOptions);
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;

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
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
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

TEST (SystemTest, randomNetwork) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    unsigned int n_reticulations = 5;
    AnnotatedNetwork ann_network = build_random_annotated_network(smallOptions, n_reticulations);
    assert(ann_network.network.num_reticulations() == n_reticulations);
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problemFillSkippedNodesRecursive) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    unsigned int n_reticulations = 2;
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "((C:0.05)#0:0.05::0.5,((B:0.05,#0:1::0.5):0.025)#1:0.025::0.5,(A:0.1,(D:0.05,#1:1::0.5):0.05):0.1);");
    assert(ann_network.network.num_reticulations() == n_reticulations);
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problemConnectSubtreeRecursive) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    unsigned int n_reticulations = 2;
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "((C:0.1,((D:0.05,((A:0.025)#1:0.025::0.5)#0:1::0.5):0.025,#1:1::0.5):0.025):0.1,B:0.1,#0:0.05::0.5);");
    assert(ann_network.network.num_reticulations() == n_reticulations);
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problemCreateOperationsUpdatedReticulation) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    unsigned int n_reticulations = 2;
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "(((C:0.05)#0:0.025::0.5,((B:0.1,(D:0.05,#0:1::0.5):0.05):0.05)#1:1::0.5):0.025,#1:0.05::0.5,A:0.1);");
    assert(ann_network.network.num_reticulations() == n_reticulations);
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problemPllUpdatePartials) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    unsigned int n_reticulations = 2;
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "(((C:0.05)#0:0.025::0.5)#1:0.025::0.5,(B:0.1,D:0.1):0.1,(A:0.05,(#0:0.5::0.5,#1:1::0.5):0.5):0.05);");
    assert(ann_network.network.num_reticulations() == n_reticulations);
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problem3) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    unsigned int n_reticulations = 3;
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "(((C:0.05)#1:0.05::0.5,(D:0.05,(#1:0.5::0.5,(A:0.025)#2:1::0.5):0.5):0.05):0.1,(B:0.05)#0:0.05::0.5,(#2:0.025::0.5,#0:1::0.5):0.05);");
    assert(ann_network.network.num_reticulations() == n_reticulations);
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problem4) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    unsigned int n_reticulations = 4;
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "(((C:0.05)#1:0.025::0.5,(#1:0.5::0.5)#2:1::0.5):0.025,(((B:0.05,(A:0.05)#0:1::0.5):0.025,#2:0.5::0.5):0.025,D:0.1):0.1,#0:0.05::0.5);");
    assert(ann_network.network.num_reticulations() == n_reticulations);
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problem5) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "(((C:0.05)#1:0.025::0.5,((((B:0.1,D:0.1):0.025,#1:1::0.5):0.025)#0:0.5::0.5)#2:1::0.5):0.025,#0:0.05::0.5,(A:0.05,#2:0.5::0.5):0.05);");
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problem6) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "((((C:0.025)#1:0.0125::0.5,(D:0.05)#2:1::0.5):0.0125,((((A:0.025,#1:1::0.5):0.0125)#3:0.00625::0.5,(#2:0.025::0.5)#4:1::0.5):0.00625)#0:1::0.5):0.05,B:0.1,((#0:0.025::0.5,#3:1::0.5):0.025,#4:0.025::0.5):0.1);");
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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

TEST (SystemTest, problem7) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions,
            "(C:0.1,(((B:0.05,((((A:0.05)#0:0.025::0.5)#2:0.25::0.5)#4:0.25::0.5)#3:0.5::0.5):0.05,(((D:0.05,#0:1::0.5):0.0125,(#3:0.5::0.5,#4:1::0.5):0.5):0.0125)#1:0.025::0.5):0.05,#1:1::0.5):0.05,#2:0.025::0.5);");
    std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network.network) << "\n";

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
