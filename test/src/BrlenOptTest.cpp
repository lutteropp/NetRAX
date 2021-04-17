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

TEST (BrlenOptTest, treeVirtualRootProblem) {
    // initial setup
    std::string treePath = DATA_PATH + "0_4_taxa_1_reticulations_CELINE_PERFECT_SAMPLING_200_msasize_1_0_brlenScaler_0_5_reticulation_prob.raxml.bestTree";
    std::string msaPath = DATA_PATH + "0_4_taxa_1_reticulations_CELINE_PERFECT_SAMPLING_200_msasize_1_0_brlenScaler_0_5_reticulation_prob_msa.txt";
    std::string partitionsPath = DATA_PATH + "0_4_taxa_1_reticulations_CELINE_PERFECT_SAMPLING_200_msasize_1_0_brlenScaler_0_5_reticulation_prob_partitions.txt";
    NetraxOptions treeOptions;
    treeOptions.start_network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.model_file = partitionsPath;
    treeOptions.use_repeats = false;
    treeOptions.seed = 42;
    AnnotatedNetwork ann_network = build_annotated_network(treeOptions);
    init_annotated_network(ann_network);

    std::cout << exportDebugInfo(ann_network) << "\n";

    //optimizeModel(ann_network);

    ann_network.cached_logl_valid = false;
    double old_logl = computeLoglikelihood(ann_network, 1, 1);

    size_t pmatrix_index = 5;
    std::cout << "Testing with pmatrix index: " << pmatrix_index << "\n";

    Node* old_virtual_root = ann_network.network.root;
    Edge* edge = ann_network.network.edges_by_index[pmatrix_index];
    Node* new_virtual_root = getSource(ann_network.network, edge);
    Node* new_virtual_root_back = getTarget(ann_network.network, edge);
    std::cout << "old_virtual_root: " << old_virtual_root->clv_index << "\n";
    std::cout << "new_virtual_root: " << new_virtual_root->clv_index << "\n";
    std::cout << "new_virtual_root_back: " << new_virtual_root_back->clv_index << "\n";

    std::cout << "true logl old brlen: " << old_logl << "\n";
    std::cout << "  true logl old brlen, partition 0: " << ann_network.fake_treeinfo->partition_loglh[0] << "\n";
    std::cout << "  true logl old brlen, partition 1: " << ann_network.fake_treeinfo->partition_loglh[1] << "\n";

    auto oldTrees = extractOldTrees(ann_network, ann_network.network.root);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, ann_network.network.num_branches() - 1);

    updateCLVsVirtualRerootTrees(ann_network, old_virtual_root, new_virtual_root, new_virtual_root_back);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
    }
    ann_network.cached_logl_valid = false;


    double old_brlen = ann_network.fake_treeinfo->branch_lengths[0][pmatrix_index];
    std::cout << "old brlen: " << old_brlen << "\n";
    double old_logl_reroot = computeLoglikelihoodBrlenOpt(ann_network, oldTrees, edge->pmatrix_index, 1, 1);

    std::cout << "reroot logl old brlen: " << old_logl_reroot << "\n";
    std::cout << "  reroot logl old brlen, partition 0: " << ann_network.fake_treeinfo->partition_loglh[0] << "\n";
    std::cout << "  reroot logl old brlen, partition 1: " << ann_network.fake_treeinfo->partition_loglh[1] << "\n";
    ann_network.fake_treeinfo->branch_lengths[0][pmatrix_index] = 0.049539358966596169775442604077397845685482025146484375;
    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        ann_network.fake_treeinfo->pmatrix_valid[partition_idx][pmatrix_index] = 0;
    }
    ann_network.cached_logl_valid = false;
    double new_logl_reroot = computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index);
    std::cout << "reroot logl new brlen: " << new_logl_reroot << "\n";
    std::cout << "  reroot logl new brlen, partition 0: " << ann_network.fake_treeinfo->partition_loglh[0] << "\n";
    std::cout << "  reroot logl new brlen, partition 1: " << ann_network.fake_treeinfo->partition_loglh[1] << "\n";

    invalidatePmatrixIndex(ann_network, pmatrix_index);
    double true_logl_new_brlen = computeLoglikelihood(ann_network);
    std::cout << "true logl new brlen: " << true_logl_new_brlen << "\n";
    std::cout << "  true logl new brlen, partition 0: " << ann_network.fake_treeinfo->partition_loglh[0] << "\n";
    std::cout << "  true logl new brlen, partition 1: " << ann_network.fake_treeinfo->partition_loglh[1] << "\n";
    updateCLVsVirtualRerootTrees(ann_network, ann_network.network.root, getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]), getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]));
    ann_network.cached_logl_valid = false;
    std::cout << "reroot logl new brlen, after computing true logl new brlen: " << computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index) << "\n";
    std::cout << "  reroot logl new brlen, after computing true logl new brlen, partition 0: " << ann_network.fake_treeinfo->partition_loglh[0] << "\n";
    std::cout << "  reroot logl new brlen, after computing true logl new brlen, partition 1: " << ann_network.fake_treeinfo->partition_loglh[1] << "\n";
    ann_network.fake_treeinfo->branch_lengths[0][pmatrix_index] = old_brlen;
    //invalidatePmatrixIndex(ann_network, pmatrix_index);
    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        ann_network.fake_treeinfo->pmatrix_valid[partition_idx][pmatrix_index] = 0;
    }
    ann_network.cached_logl_valid = false;
    std::cout << "reroot logl old brlen here: " << computeLoglikelihoodBrlenOpt(ann_network, oldTrees, pmatrix_index) << "\n";
    std::cout << "  reroot logl old brlen here, partition 0: " << ann_network.fake_treeinfo->partition_loglh[0] << "\n";
    std::cout << "  reroot logl old brlen here, partition 1: " << ann_network.fake_treeinfo->partition_loglh[1] << "\n";
    invalidatePmatrixIndex(ann_network, pmatrix_index);
    double true_logl_old_brlen_again = computeLoglikelihood(ann_network);
    std::cout << "true logl old brlen again: " << true_logl_old_brlen_again << "\n";
    std::cout << "  true logl old brlen again, partition 0: " << ann_network.fake_treeinfo->partition_loglh[0] << "\n";
    std::cout << "  true logl old brlen again, partition 1: " << ann_network.fake_treeinfo->partition_loglh[1] << "\n";

    ASSERT_DOUBLE_EQ(old_logl, old_logl_reroot);
    ASSERT_DOUBLE_EQ(new_logl_reroot, true_logl_new_brlen);
}

TEST (BrlenOptTest, treeVirtualRoots) {
    // initial setup
    std::string treePath = DATA_PATH + "0_4_taxa_1_reticulations_CELINE_PERFECT_SAMPLING_200_msasize_1_0_brlenScaler_0_5_reticulation_prob.raxml.bestTree";
    std::string msaPath = DATA_PATH + "0_4_taxa_1_reticulations_CELINE_PERFECT_SAMPLING_200_msasize_1_0_brlenScaler_0_5_reticulation_prob_msa.txt";
    //std::string partitionsPath = DATA_PATH + "0_4_taxa_1_reticulations_CELINE_PERFECT_SAMPLING_200_msasize_1_0_brlenScaler_0_5_reticulation_prob_partitions.txt";
    NetraxOptions treeOptions;
    treeOptions.start_network_file = treePath;
    treeOptions.msa_file = msaPath;
    //treeOptions.model_file = partitionsPath;
    treeOptions.use_repeats = false;
    treeOptions.seed = 42;
    AnnotatedNetwork annTreeNetwork = build_annotated_network(treeOptions);
    init_annotated_network(annTreeNetwork);
    annTreeNetwork.cached_logl_valid = false;
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

        for (size_t p = 0; p < annTreeNetwork.fake_treeinfo->partition_count; ++p) {
            annTreeNetwork.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
        annTreeNetwork.cached_logl_valid = false;
        double new_logl = computeLoglikelihoodBrlenOpt(annTreeNetwork, oldTrees, edge->pmatrix_index, 1, 1);

        ASSERT_DOUBLE_EQ(old_logl, new_logl);

        // change the brlen
        double old_brlen = annTreeNetwork.fake_treeinfo->branch_lengths[0][pmatrix_index];
        annTreeNetwork.fake_treeinfo->branch_lengths[0][pmatrix_index] = 12345;
        for (size_t p = 0; p < annTreeNetwork.fake_treeinfo->partition_count; ++p) {
            annTreeNetwork.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
        annTreeNetwork.cached_logl_valid = false;
        double mid_logl = computeLoglikelihoodBrlenOpt(annTreeNetwork, oldTrees, edge->pmatrix_index, 1, 1);
        ASSERT_TRUE(mid_logl != new_logl);

        // take back the changed brlen
        annTreeNetwork.fake_treeinfo->branch_lengths[0][pmatrix_index] = old_brlen;
        for (size_t p = 0; p < annTreeNetwork.fake_treeinfo->partition_count; ++p) {
            annTreeNetwork.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
        annTreeNetwork.cached_logl_valid = false;
        double recomputed_logl = computeLoglikelihoodBrlenOpt(annTreeNetwork, oldTrees, edge->pmatrix_index, 1, 1);

        ASSERT_DOUBLE_EQ(old_logl, recomputed_logl);

        // reroot the network
        invalidatePmatrixIndex(annTreeNetwork, pmatrix_index);
        computeLoglikelihood(annTreeNetwork);

        oldTrees = extractOldTrees(annTreeNetwork, old_virtual_root);
        //old_virtual_root = new_virtual_root;
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
        size_t n_trees = annTreeNetwork.pernode_displayed_tree_data[annTreeNetwork.network.root->clv_index].num_active_displayed_trees;
        for (size_t i = 0; i < n_trees; ++i) {
                DisplayedTreeData& tree = annTreeNetwork.pernode_displayed_tree_data[annTreeNetwork.network.root->clv_index].displayed_trees[i];
                std::cout << "correct partition " << p << ", tree " << i << " logl: " << tree.treeLoglData[p].tree_logl << ", logprob: " << tree.treeLoglData[p].tree_logprob << "\n";
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
