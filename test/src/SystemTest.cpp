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
#include "src/optimization/Moves.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>

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
    treeOptions.start_network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = true;
    const RaxmlInstance instance = createRaxmlInstance(treeOptions);
    //treeWrapper.enableRaxmlDebugOutput();
    TreeInfo *info = createRaxmlTreeinfo(normalTree.pll_utree_copy(), instance);

    // initial logl computation
    double initial_logl = info->loglh(false);
    std::cout << "Initial loglikelihood: " << initial_logl << "\n";

    // model parameter optimization
    double modelopt_logl = info->optimize_model(instance.opts.lh_epsilon);
    std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

    std::cout << "The branch lengths before brlen optimization are:\n";
    for (size_t i = 0; i < info->pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << info->pll_treeinfo().branch_lengths[0][i]
                << "\n";
    }

    // branch length optimization
    double brlenopt_logl = info->optimize_branches(instance.opts.lh_epsilon, 1);
    std::cout << "Loglikelihood after branch length optimization: " << brlenopt_logl << "\n";

    std::cout << "The optimized branch lengths are:\n";
    for (size_t i = 0; i < info->pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << info->pll_treeinfo().branch_lengths[0][i]
                << "\n";
    }

    // model parameter optimization
    double modelopt2_logl = info->optimize_model(instance.opts.lh_epsilon);
    std::cout << "Loglikelihood after model optimization again: " << modelopt2_logl << "\n";

    delete info;
}

void completeRun(AnnotatedNetwork &ann_network) {
    //std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network) << "\n";
    throw std::runtime_error("TODO: Update this test function");
    //optimizeEverything(ann_network);
    //std::cout << exportDebugInfo(ann_network.network) << "\n";
    std::cout << toExtendedNewick(ann_network) << "\n";
}

TEST (SystemTest, allTree) {
    // initial setup
    std::string treePath = DATA_PATH + "tree.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions treeOptions;
    treeOptions.start_network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = true;

    const RaxmlInstance instance = createRaxmlInstance(treeOptions);
    AnnotatedNetwork ann_network = build_annotated_network(treeOptions, instance);
    init_annotated_network(ann_network);
    completeRun(ann_network);
}

TEST (SystemTest, allNetwork) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.start_network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    const RaxmlInstance instance = createRaxmlInstance(smallOptions);
    AnnotatedNetwork ann_network = build_annotated_network(smallOptions, instance);
    init_annotated_network(ann_network);

    completeRun(ann_network);
}

TEST (SystemTest, randomNetwork) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.start_network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    unsigned int n_reticulations = 8;
    const RaxmlInstance instance = createRaxmlInstance(smallOptions);
    AnnotatedNetwork ann_network = build_random_annotated_network(smallOptions, instance, rand());
    init_annotated_network(ann_network);
    add_extra_reticulations(ann_network, n_reticulations);
    assert(ann_network.network.num_reticulations() == n_reticulations);

    completeRun(ann_network);
}

void problemTest(const std::string &newick) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.start_network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    const RaxmlInstance instance = createRaxmlInstance(smallOptions);
    AnnotatedNetwork ann_network = build_annotated_network_from_string(smallOptions, instance, newick);
    init_annotated_network(ann_network);

    completeRun(ann_network);
}

TEST (SystemTest, problemFillSkippedNodesRecursive) {
    problemTest(
            "((C:0.05)#0:0.05::0.5,((B:0.05,#0:1::0.5):0.025)#1:0.025::0.5,(A:0.1,(D:0.05,#1:1::0.5):0.05):0.1);");
}

TEST (SystemTest, problemConnectSubtreeRecursive) {
    problemTest(
            "((C:0.1,((D:0.05,((A:0.025)#1:0.025::0.5)#0:1::0.5):0.025,#1:1::0.5):0.025):0.1,B:0.1,#0:0.05::0.5);");
}

TEST (SystemTest, problemCreateOperationsUpdatedReticulation) {
    problemTest(
            "(((C:0.05)#0:0.025::0.5,((B:0.1,(D:0.05,#0:1::0.5):0.05):0.05)#1:1::0.5):0.025,#1:0.05::0.5,A:0.1);");
}

TEST (SystemTest, problemPllUpdatePartials) {
    problemTest(
            "(((C:0.05)#0:0.025::0.5)#1:0.025::0.5,(B:0.1,D:0.1):0.1,(A:0.05,(#0:0.5::0.5,#1:1::0.5):0.5):0.05);");
}

TEST (SystemTest, problem3) {
    problemTest(
            "(((C:0.05)#1:0.05::0.5,(D:0.05,(#1:0.5::0.5,(A:0.025)#2:1::0.5):0.5):0.05):0.1,(B:0.05)#0:0.05::0.5,(#2:0.025::0.5,#0:1::0.5):0.05);");
}

TEST (SystemTest, problem4) {
    problemTest(
            "(((C:0.05)#1:0.025::0.5,(#1:0.5::0.5)#2:1::0.5):0.025,(((B:0.05,(A:0.05)#0:1::0.5):0.025,#2:0.5::0.5):0.025,D:0.1):0.1,#0:0.05::0.5);");
}

TEST (SystemTest, problem5) {
    problemTest(
            "(((C:0.05)#1:0.025::0.5,((((B:0.1,D:0.1):0.025,#1:1::0.5):0.025)#0:0.5::0.5)#2:1::0.5):0.025,#0:0.05::0.5,(A:0.05,#2:0.5::0.5):0.05);");
}

TEST (SystemTest, problem6) {
    problemTest(
            "((((C:0.025)#1:0.0125::0.5,(D:0.05)#2:1::0.5):0.0125,((((A:0.025,#1:1::0.5):0.0125)#3:0.00625::0.5,(#2:0.025::0.5)#4:1::0.5):0.00625)#0:1::0.5):0.05,B:0.1,((#0:0.025::0.5,#3:1::0.5):0.025,#4:0.025::0.5):0.1);");
}

TEST (SystemTest, problem7) {
    problemTest(
            "(C:0.1,(((B:0.05,((((A:0.05)#0:0.025::0.5)#2:0.25::0.5)#4:0.25::0.5)#3:0.5::0.5):0.05,(((D:0.05,#0:1::0.5):0.0125,(#3:0.5::0.5,#4:1::0.5):0.5):0.0125)#1:0.025::0.5):0.05,#1:1::0.5):0.05,#2:0.025::0.5);");
}

TEST (SystemTest, problem8) {
    problemTest(
            "((((C:0.05,((((B:0.025)#1:0.025::0.5)#0:0.5::0.5,(#1:0.5::0.5)#3:0.5::0.5):0.125)#4:1::0.5):0.05,(D:0.05,(A:0.05)#2:1::0.5):0.05):0.05,(#4:0.125::0.5,#3:1::0.5):0.25):0.05,#0:0.05::0.5,#2:0.05::0.5);");
}

TEST (SystemTest, problem9) {
    problemTest(
            "((((C:0.05)#0:0.0125::0.5,((((B:0.025)#5:0.025::0.5)#1:0.0125::0.5,(((((A:0.05,#1:1::0.5):0.05,D:0.1):0.05,(#0:0.5::0.5,#5:1::0.5):0.5):0.025)#2:0.0125::0.5)#4:1::0.5):0.0125)#3:1::0.5):0.0125,#2:1::0.5):0.025,#3:0.025::0.5,#4:0.0125::0.5);");
}

TEST (SystemTest, problem10) {
    problemTest(
            "(((((C:0.025,(((D:0.05)#2:0.0125::0.5)#4:0.0125::0.5)#3:1::0.5):0.0125,(#3:0.00625::0.5)#6:1::0.5):0.0125,((A:0.025)#1:0.025::0.5)#0:1::0.5):0.025)#5:0.025::0.5,((B:0.05,(#1:0.5::0.5,(#2:0.5::0.5,#5:1::0.5):0.5):0.5):0.05,(#6:0.00625::0.5,#4:1::0.5):0.0125):0.1,#0:0.05::0.5);");
}

TEST (SystemTest, problem11) {
    problemTest(
            "(C:0.1,((B:0.05,(((A:0.1,D:0.1):0.025)#1:0.025::0.5)#0:1::0.5):0.025,#1:1::0.5):0.025,#0:0.05::0.5);");
}

TEST (SystemTest, problem12) {
    //problemTestOptTopology("((A:1.49622e-06,(D:1.43824e-06)#0:1.28508e-06::0.5):10.0005,(#0:83.9918::0.5,B:0.130942):0.0141053,C:0.444893);", MoveType::RSPR1Move);
    problemTest("(C:0.1,(B:0.05,(A:0.05)#0:1::0.5):0.05,(#0:0.05::0.5,D:0.1):0.1);");
}

TEST (SystemTest, problem14) {
    problemTest(
            "(C:0.1,((B:0.1,((D:0.05)#0:0.025::0.5,((A:0.025)#1:0.5::0.5,(#0:0.5::0.5)#3:1::0.5):0.5):0.025):0.05)#2:0.05::0.5,((#1:0.0125::0.5,#2:1::0.5):0.0125,#3:0.5::0.5):0.05);");
}

TEST (SystemTest, problem16) {
    problemTest(
            "(((C:0.05,(((A:0.05)#0:0.5::0.5)#1:0.5::0.5)#2:1::0.5):0.025)#3:0.025::0.5,((((B:0.05,#2:0.5::0.5):0.05,D:0.1):0.025,#3:1::0.5):0.025,#1:0.5::0.5):0.05,#0:0.05::0.5);");
}

TEST (SystemTest, problem17) {
    problemTest(
            "((((((B:46.8241,C:10.0945):10.0945,(D:19.9874,A:19.8685):20.0288):20.0961)#1:53.2556::0.5)#0:15.4048::0.5)#2:44.4354::0.5,#0:27.9897::0.5,(#2:15.4048::0.5,#1:20.0961::0.5):12.6721);");
}

TEST (SystemTest, problem18) {
    problemTest(
            "(((((C:0.025)#1:0.025::0.5,(((B:0.0125)#2:0.0125::0.5,#1:1::0.5):0.025)#0:1::0.5):0.0125)#4:0.0125::0.5)#3:0.025::0.5,((#0:0.05::0.5,D:0.1):0.05,(#2:0.5::0.5,(#3:0.25::0.5)#5:1::0.5):0.5):0.05,(A:0.05,(#5:0.25::0.5,#4:1::0.5):0.5):0.05);");
}

TEST (SystemTest, problem19) {
    problemTest(
            "((C:0.05)#1:0.05::0.5,((B:0.1,((((D:0.05)#0:0.00625::0.5,(#1:0.25::0.5)#4:1::0.5):0.00625)#2:0.0125::0.5,(#4:0.25::0.5,#2:1::0.5):0.5):0.025):0.05,(#0:0.5::0.5)#3:0.5::0.5):0.05,(A:0.05,#3:1::0.5):0.05);");
}

TEST (SystemTest, problem20) {
    problemTest(
            "((((((((C:0.025,((((A:0.025)#3:0.0125::0.5)#5:0.0125::0.5)#2:0.5::0.5)#4:0.5::0.5):0.0125,#3:1::0.5):0.00625,#4:1::0.5):0.00625)#0:0.025::0.5)#1:0.025::0.5,(D:0.05,#5:1::0.5):0.05):0.05,#1:1::0.5):0.05,(B:0.05,#0:1::0.5):0.05,#2:0.05::0.5);");
}
