/*
 * BrlenOptTest.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: sarah
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"
#include "src/Api.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/graph/Common.hpp"
#include "src/utils.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "../../examples/sample_networks/";

TEST (BrlenOptTest, testTheTest) {
    ASSERT_TRUE(true);
}

TEST (BrlenOptTest, tree) {
    // initial setup
    std::string treePath = DATA_PATH + "tree.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.nw";
    NetraxOptions treeOptions;
    treeOptions.network_file = treePath;
    treeOptions.msa_file = msaPath;
    treeOptions.use_repeats = true;
    RaxmlWrapper treeWrapper = RaxmlWrapper(treeOptions);
    //treeWrapper.enableRaxmlDebugOutput();

    Tree normalTree = Tree::loadFromFile(treePath);
    TreeInfo* infoRaxml = treeWrapper.createRaxmlTreeinfo(normalTree.pll_utree_copy());
    AnnotatedNetwork annTreeNetwork = build_annotated_network(treeOptions);
    TreeInfo* infoNetwork = annTreeNetwork.raxml_treeinfo.get();

    // initial logl computation
    double initial_logl_raxml = infoRaxml->loglh(false);
    std::cout << "RAXML - Initial loglikelihood: " << initial_logl_raxml << "\n";
    double initial_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";
    ASSERT_FLOAT_EQ(initial_logl_raxml, initial_logl_network);

    ASSERT_EQ(infoRaxml->pll_treeinfo().tree->edge_count + 1, infoNetwork->pll_treeinfo().tree->edge_count);

    // Compute branch id mapping to check the branch lengths
    std::vector<std::vector<size_t> > utreeIDToNetworkID = getDtBranchToNetworkBranchMapping(
            *infoRaxml->pll_treeinfo().tree, annTreeNetwork.network, 0);

    std::cout << "utreeIDToNetworkID.size(): " << utreeIDToNetworkID.size() << "\n";

    std::cout << "RAXML - The branch lengths before brlen optimization are:\n";
    for (size_t i = 0; i < infoRaxml->pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << infoRaxml->pll_treeinfo().branch_lengths[0][i] << "\n";
    }
    std::cout << "NETWORK - The branch lengths before brlen optimization are:\n";
    for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
        if (utreeIDToNetworkID[i].size() == 1) {
            std::cout << " " << std::setprecision(17)
                    << infoNetwork->pll_treeinfo().branch_lengths[0][utreeIDToNetworkID[i][0]] << "\n";
        }
    }

    for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
        if (utreeIDToNetworkID[i].size() == 1) {
            ASSERT_FLOAT_EQ(infoRaxml->pll_treeinfo().branch_lengths[0][i],
                    infoNetwork->pll_treeinfo().branch_lengths[0][utreeIDToNetworkID[i][0]]);
        }
    }

    // branch length optimization
    double brlenopt_logl_raxml = infoRaxml->optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
    std::cout << "RAXML - Loglikelihood after branch length optimization: " << brlenopt_logl_raxml << "\n";
    double brlenopt_logl_network = infoNetwork->optimize_branches(treeWrapper.getRaxmlOptions().lh_epsilon, 1);
    std::cout << "NETWORK - Loglikelihood after branch length optimization: " << brlenopt_logl_network << "\n";

    std::cout << "RAXML - The optimized branch lengths are:\n";
    for (size_t i = 0; i < infoRaxml->pll_treeinfo().tree->edge_count; ++i) {
        std::cout << " " << std::setprecision(17) << infoRaxml->pll_treeinfo().branch_lengths[0][i] << "\n";
    }
    std::cout << "NETWORK - The optimized branch lengths are:\n";
    for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
        std::cout << " " << std::setprecision(17) << infoNetwork->pll_treeinfo().branch_lengths[0][i] << "\n";
    }
    // extract the actual network data structure
    Network *network_ptr = &annTreeNetwork.network;
    std::cout << "NETWORK - The ACTUAL optimized branch lengths are:\n";
    for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
        std::cout << std::setprecision(17) << " pmatrix_idx = " << network_ptr->edges[i].pmatrix_index << " -> brlen = "
                << network_ptr->edges[i].length << "\n";
    }

    for (size_t i = 0; i < utreeIDToNetworkID.size(); ++i) {
        if (utreeIDToNetworkID[i].size() == 1) {
            ASSERT_FLOAT_EQ(infoRaxml->pll_treeinfo().branch_lengths[0][i],
                    infoNetwork->pll_treeinfo().branch_lengths[0][utreeIDToNetworkID[i][0]]);
        }
    }
    double normal_logl_raxml = infoRaxml->loglh(0);
    double normal_logl_network = infoNetwork->loglh(0);
    std::cout << "RAXML - Loglikelihood when called normally: " << normal_logl_raxml << "\n";
    std::cout << "NETWORK - Loglikelihood when called normally: " << normal_logl_network << "\n";

    ASSERT_FLOAT_EQ(brlenopt_logl_network, normal_logl_network);

    ASSERT_FLOAT_EQ(brlenopt_logl_raxml, brlenopt_logl_network);

    delete infoRaxml;
}

TEST (BrlenOptTest, small) {
    // initial setup
    std::string smallPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.nw";
    NetraxOptions smallOptions;
    smallOptions.network_file = smallPath;
    smallOptions.msa_file = msaPath;
    smallOptions.use_repeats = true;
    RaxmlWrapper smallWrapper = RaxmlWrapper(smallOptions);
    //smallWrapper.enableRaxmlDebugOutput();

    AnnotatedNetwork annTreeNetwork = build_annotated_network(smallOptions);

    // initial logl computation
    double initial_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";

    // branch length optimization
    double brlenopt_logl_network = optimizeBranches(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood after branch length optimization: " << brlenopt_logl_network << "\n";

    double normal_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood when called normally: " << normal_logl_network << "\n";

    ASSERT_FLOAT_EQ(brlenopt_logl_network, normal_logl_network);
}

TEST (BrlenOptTest, celineFake) {
    // initial setup
    std::string celinePath = DATA_PATH + "celine.nw";
    std::string msaPath = DATA_PATH + "celine_fake_alignment.txt";
    NetraxOptions celineOptions;
    celineOptions.network_file = celinePath;
    celineOptions.msa_file = msaPath;
    celineOptions.use_repeats = true;
    RaxmlWrapper celineWrapper = RaxmlWrapper(celineOptions);
    //smallWrapper.enableRaxmlDebugOutput();

    AnnotatedNetwork annTreeNetwork = build_annotated_network(celineOptions);

    // initial logl computation
    double initial_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";

    // branch length optimization
    double brlenopt_logl_network = optimizeBranches(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood after branch length optimization: " << brlenopt_logl_network << "\n";

    double normal_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood when called normally: " << normal_logl_network << "\n";

    ASSERT_FLOAT_EQ(brlenopt_logl_network, normal_logl_network);
}

TEST (BrlenOptTest, celineFakeWithModelopt) {
    // initial setup
    std::string celinePath = DATA_PATH + "celine.nw";
    std::string msaPath = DATA_PATH + "celine_fake_alignment.txt";
    NetraxOptions celineOptions;
    celineOptions.network_file = celinePath;
    celineOptions.msa_file = msaPath;
    celineOptions.use_repeats = true;
    RaxmlWrapper celineWrapper = RaxmlWrapper(celineOptions);
    //smallWrapper.enableRaxmlDebugOutput();
    AnnotatedNetwork annTreeNetwork = build_annotated_network(celineOptions);

    // initial logl computation
    double initial_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Initial loglikelihood: " << initial_logl_network << "\n";

    // model parameter optimization
    double modelopt_logl = optimizeModel(annTreeNetwork);
    std::cout << "Loglikelihood after model optimization: " << modelopt_logl << "\n";

    std::cout << "The entire network would like these model params:\n";
    print_model_params(*annTreeNetwork.fake_treeinfo);
    std::cout << "\n";

    // branch length optimization
    double brlenopt_logl_network = optimizeBranches(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood after branch length optimization: " << brlenopt_logl_network << "\n";

    double normal_logl_network = computeLoglikelihood(annTreeNetwork);
    std::cout << "NETWORK - Loglikelihood when called normally: " << normal_logl_network << "\n";

    ASSERT_FLOAT_EQ(brlenopt_logl_network, normal_logl_network);

    // print the network with brlen variance as branch support values
    std::cout << "network with brlen variance as branch support values:\n" << toExtendedNewick(annTreeNetwork.network) << "\n";
}
