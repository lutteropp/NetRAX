/*
 * BrlenOptTest.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: sarah
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/optimization/BranchLengthOptimization.hpp"
#include "src/optimization/ModelOptimization.hpp"
#include "src/graph/NetworkFunctions.hpp"
#include "src/graph/NetworkTopology.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"
#include "src/DebugPrintFunctions.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/utils.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "examples/sample_networks/";

TEST (BrlenOptTest, testTheTest) {
    ASSERT_TRUE(true);
}

TEST (BrlenOptTest, treeVirtualRoots) {
    // initial setup
    std::string treePath = DATA_PATH + "tree.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions treeOptions;
    treeOptions.start_network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = false;
    treeOptions.seed = 42;
    AnnotatedNetwork annTreeNetwork = build_annotated_network(treeOptions);
    init_annotated_network(annTreeNetwork);
    double old_logl = computeLoglikelihood(annTreeNetwork, 1, 1);

    std::cout << exportDebugInfo(annTreeNetwork) << "\n";

    Node* old_virtual_root = annTreeNetwork.network.root;
    auto oldTrees = extractOldTrees(annTreeNetwork, annTreeNetwork.network.root);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, annTreeNetwork.network.num_branches() - 1);
    for (size_t i = 0; i < 100; ++i) {
        size_t pmatrix_index = dist(annTreeNetwork.rng);
        std::cout << "Testing with pmatrix index: " << pmatrix_index << "\n";

        Edge* edge = annTreeNetwork.network.edges_by_index[pmatrix_index];
        Node* new_virtual_root = getSource(annTreeNetwork.network, edge);
        Node* new_virtual_root_back = getTarget(annTreeNetwork.network, edge);
        std::cout << "old_virtual_root: " << old_virtual_root->clv_index << "\n";
        std::cout << "new_virtual_root: " << new_virtual_root->clv_index << "\n";
        std::cout << "new_virtual_root_back: " << new_virtual_root_back->clv_index << "\n";
        updateCLVsVirtualRerootTrees(annTreeNetwork, old_virtual_root, new_virtual_root, new_virtual_root_back);
        double new_logl = computeLoglikelihoodBrlenOpt(annTreeNetwork, oldTrees, edge->pmatrix_index, 1, 1);

        ASSERT_DOUBLE_EQ(old_logl, new_logl);

        oldTrees = extractOldTrees(annTreeNetwork, old_virtual_root);
        old_virtual_root = new_virtual_root;
    }
}

TEST (BrlenOptTest, small) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.start_network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    //smallWrapper.enableRaxmlDebugOutput();
    AnnotatedNetwork annTreeNetwork = build_annotated_network(smallOptions);
    init_annotated_network(annTreeNetwork);

    // initial logl computation
    double initial_logl_network = computeLoglikelihood(annTreeNetwork, 1, 1);
    std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";

    // branch length optimization
    optimizeBranches(annTreeNetwork);
    double brlenopt_logl_network = computeLoglikelihood(annTreeNetwork, 1, 1);
    std::cout << "NETWORK - Loglikelihood after branch length optimization: "
            << brlenopt_logl_network << "\n";
}

TEST (BrlenOptTest, smallVirtualRoots) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";
    NetraxOptions smallOptions;
    smallOptions.start_network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = false;
    smallOptions.seed = 42;
    AnnotatedNetwork annTreeNetwork = build_annotated_network(smallOptions);
    init_annotated_network(annTreeNetwork);
    double old_logl = computeLoglikelihood(annTreeNetwork, 1, 1);

    for (size_t p = 0; p < annTreeNetwork.fake_treeinfo->partition_count; ++p) {
        size_t n_trees = annTreeNetwork.pernode_displayed_tree_data[p][annTreeNetwork.network.root->clv_index].num_active_displayed_trees;
        for (size_t i = 0; i < n_trees; ++i) {
                DisplayedTreeData& tree = annTreeNetwork.pernode_displayed_tree_data[p][annTreeNetwork.network.root->clv_index].displayed_trees[i];
                std::cout << "correct partition " << p << ", tree " << i << " logl: " << tree.treeLoglData.tree_logl << ", logprob: " << tree.treeLoglData.tree_logprob << "\n";
        }
    }

    std::cout << exportDebugInfo(annTreeNetwork) << "\n";

    Node* old_virtual_root = annTreeNetwork.network.root;
    auto oldTrees = extractOldTrees(annTreeNetwork, annTreeNetwork.network.root);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, annTreeNetwork.network.num_branches() - 1);
    for (size_t i = 0; i < annTreeNetwork.network.num_nodes(); ++i) {
        size_t pmatrix_index = 8;//i;//dist(annTreeNetwork.rng);
        std::cout << "Testing with pmatrix index: " << pmatrix_index << "\n";

        Edge* edge = annTreeNetwork.network.edges_by_index[pmatrix_index];
        Node* new_virtual_root = getSource(annTreeNetwork.network, edge);
        Node* new_virtual_root_back = getTarget(annTreeNetwork.network, edge);
        std::cout << "old_virtual_root: " << old_virtual_root->clv_index << "\n";
        std::cout << "new_virtual_root: " << new_virtual_root->clv_index << "\n";
        std::cout << "new_virtual_root_back: " << new_virtual_root_back->clv_index << "\n";
        updateCLVsVirtualRerootTrees(annTreeNetwork, old_virtual_root, new_virtual_root, new_virtual_root_back);
        double new_logl = computeLoglikelihoodBrlenOpt(annTreeNetwork, oldTrees, edge->pmatrix_index, 1, 1);

        ASSERT_DOUBLE_EQ(old_logl, new_logl);

        oldTrees = extractOldTrees(annTreeNetwork, old_virtual_root);

        for (size_t p = 0; p < annTreeNetwork.fake_treeinfo->partition_count; ++p) {
            invalidateHigherCLVs(annTreeNetwork, new_virtual_root, p, true);
        }
        double recomputedLogl = computeLoglikelihood(annTreeNetwork, 1, 0);
        oldTrees = extractOldTrees(annTreeNetwork, annTreeNetwork.network.root);
        ASSERT_DOUBLE_EQ(old_logl, recomputedLogl);

        //old_virtual_root = new_virtual_root;
    }
}

TEST (BrlenOptTest, celineFake) {
    // initial setup
    std::string celinePath = DATA_PATH + "celine.nw";
    std::string msaPath = DATA_PATH + "celine_fake_alignment.txt";
    NetraxOptions celineOptions;
    celineOptions.start_network_file = celinePath;
    celineOptions.msa_file = msaPath;
    celineOptions.use_repeats = true;
    RaxmlWrapper celineWrapper = RaxmlWrapper(celineOptions);
    //smallWrapper.enableRaxmlDebugOutput();

    AnnotatedNetwork annTreeNetwork = build_annotated_network(celineOptions);
    init_annotated_network(annTreeNetwork);

    // initial logl computation
    double initial_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";

    // branch length optimization
    optimizeBranches(annTreeNetwork);
    double brlenopt_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood after branch length optimization: "
            << brlenopt_logl_network << "\n";
}

TEST (BrlenOptTest, celineFakeWithModelopt) {
    // initial setup
    std::string celinePath = DATA_PATH + "celine.nw";
    std::string msaPath = DATA_PATH + "celine_fake_alignment.txt";
    NetraxOptions celineOptions;
    celineOptions.start_network_file = celinePath;
    celineOptions.msa_file = msaPath;
    celineOptions.use_repeats = true;
    RaxmlWrapper celineWrapper = RaxmlWrapper(celineOptions);
    //smallWrapper.enableRaxmlDebugOutput();
    AnnotatedNetwork annTreeNetwork = build_annotated_network(celineOptions);
    init_annotated_network(annTreeNetwork);

    // initial logl computation
    double initial_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";

    // model parameter optimization
    optimizeModel(annTreeNetwork);
    double modelopt_logl = computeLoglikelihood(annTreeNetwork);
    std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

    std::cout << "The entire network would like these model params:\n";
    print_model_params(*annTreeNetwork.fake_treeinfo);
    std::cout << "\n";

    // branch length optimization
    optimizeBranches(annTreeNetwork);
    double brlenopt_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood after branch length optimization: "
            << brlenopt_logl_network << "\n";

    // print the network with brlen variance as branch support values
    std::cout << "network with brlen variance as branch support values:\n"
            << toExtendedNewick(annTreeNetwork) << "\n";
}
