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

#include "src/NetworkInfo.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "examples/sample_networks/";

void randomNNIMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetworkInfo networkinfo = buildNetworkInfo(networkPath, msaPath, useRepeats);

    double initial_logl = loglh(networkinfo);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < networkinfo.network.edges.size(); ++i) {
        std::vector<RNNIMove> candidates = possibleRNNIMoves(networkinfo.network, networkinfo.network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            performMove(networkinfo.network, candidates[j]);
            double moved_logl = loglh(networkinfo);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            undoMove(networkinfo.network, candidates[j]);
            double back_logl = loglh(networkinfo);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void randomSPRMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetworkInfo networkinfo = buildNetworkInfo(networkPath, msaPath, useRepeats);

    double initial_logl = loglh(networkinfo);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < networkinfo.network.edges.size(); ++i) {
        std::vector<RSPRMove> candidates = possibleRSPRMoves(networkinfo.network, networkinfo.network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            performMove(networkinfo.network, candidates[j]);
            double moved_logl = loglh(networkinfo);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            undoMove(networkinfo.network, candidates[j]);
            double back_logl = loglh(networkinfo);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

TEST (MovesTest, nniCeline) {
    randomNNIMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}
TEST (MovesTest, nniSmall) {
    randomNNIMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}
TEST (MovesTest, sprCeline) {
    randomSPRMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}
TEST (MovesTest, sprSmall) {
    randomSPRMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

