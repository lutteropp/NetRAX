/*
 * MovesTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/graph/Moves.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"

#include "src/graph/NetworkFunctions.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/graph/Common.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "examples/sample_networks/";

void randomNNIMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    Network network = netrax::readNetworkFromFile(networkPath);
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    RaxmlWrapper wrapper(options);
    TreeInfo network_treeinfo = wrapper.createRaxmlTreeinfo(network);
    RaxmlWrapper::NetworkParams *params =
            (RaxmlWrapper::NetworkParams*) network_treeinfo.pll_treeinfo().likelihood_computation_params;
    pllmod_treeinfo_t treeinfo = *(params->network_treeinfo);

    double initial_logl = computeLoglikelihood(network, treeinfo, 0, 1);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.edges.size(); ++i) {
        std::vector<RNNIMove> candidates = possibleRNNIMoves(network, network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            performMove(network, candidates[j]);
            double moved_logl = computeLoglikelihood(network, treeinfo, 0, 1);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            undoMove(network, candidates[j]);
            double back_logl = computeLoglikelihood(network, treeinfo, 0, 1);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void randomSPRMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    Network network = netrax::readNetworkFromFile(networkPath);
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    RaxmlWrapper wrapper(options);
    TreeInfo network_treeinfo = wrapper.createRaxmlTreeinfo(network);
    RaxmlWrapper::NetworkParams *params =
            (RaxmlWrapper::NetworkParams*) network_treeinfo.pll_treeinfo().likelihood_computation_params;
    pllmod_treeinfo_t treeinfo = *(params->network_treeinfo);

    double initial_logl = computeLoglikelihood(network, treeinfo, 0, 1);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.edges.size(); ++i) {
        std::vector<RSPRMove> candidates = possibleRSPRMoves(network, network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            performMove(network, candidates[j]);
            double moved_logl = computeLoglikelihood(network, treeinfo, 0, 1);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            undoMove(network, candidates[j]);
            double back_logl = computeLoglikelihood(network, treeinfo, 0, 1);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

TEST (MovesTest, nniSmall) {
    randomNNIMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, nniCeline) {
    randomNNIMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, sprSmall) {
    randomSPRMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, sprCeline) {
    randomSPRMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}


