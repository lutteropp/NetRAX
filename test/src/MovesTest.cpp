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
#include "src/Api.hpp"

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
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    Network &network = ann_network.network;
    std::string initialDebugInfo = exportDebugInfo(network);
    std::cout << initialDebugInfo;

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RNNIMove> candidates = possibleRNNIMoves(ann_network, &network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            //std::cout << i << ", " << j << "\n";
            //if (i != 35 || j != 3) {
            //    continue;
            //}
            std::cout << "perform " << toString(candidates[j]);
            performMove(ann_network, candidates[j]);
            double moved_logl = computeLoglikelihood(ann_network);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            std::cout << "logl after move: " << moved_logl << "\n";
            std::cout << "undo " << toString(candidates[j]) << "\n";
            undoMove(ann_network, candidates[j]);
            std::string debugInfoAfterUndo = exportDebugInfo(network);
            EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
            double back_logl = computeLoglikelihood(ann_network);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void randomSPRMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    Network &network = ann_network.network;
    std::string initialDebugInfo = exportDebugInfo(network);
    std::cout << initialDebugInfo;

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> candidates = possibleRSPRMoves(ann_network, &network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            std::cout << "perform " << toString(candidates[j]);
            performMove(ann_network, candidates[j]);
            double moved_logl = computeLoglikelihood(ann_network);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            std::cout << "logl after move: " << moved_logl << "\n";
            std::cout << "undo " << toString(candidates[j]) << "\n";
            undoMove(ann_network, candidates[j]);
            std::string debugInfoAfterUndo = exportDebugInfo(network);
            EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
            double back_logl = computeLoglikelihood(ann_network);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void randomSPR1Moves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    Network &network = ann_network.network;
    std::string initialDebugInfo = exportDebugInfo(network);
    std::cout << initialDebugInfo;

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> candidates = possibleRSPR1Moves(ann_network, &network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            std::cout << "perform " << toString(candidates[j]);
            performMove(ann_network, candidates[j]);
            double moved_logl = computeLoglikelihood(ann_network);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            std::cout << "logl after move: " << moved_logl << "\n";
            std::cout << "undo " << toString(candidates[j]) << "\n";
            undoMove(ann_network, candidates[j]);
            std::string debugInfoAfterUndo = exportDebugInfo(network);
            EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
            double back_logl = computeLoglikelihood(ann_network);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void randomArcInsertionMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    Network &network = ann_network.network;
    std::string initialDebugInfo = exportDebugInfo(network);
    std::cout << initialDebugInfo;

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<ArcInsertionMove> candidates = possibleArcInsertionMoves(ann_network, &network.edges[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            std::cout << "perform " << toString(candidates[j]);
            performMove(ann_network, candidates[j]);

            //std::cout << "network after move:\n";
            //std::cout << exportDebugInfo(network) << "\n";

            double moved_logl = computeLoglikelihood(ann_network);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            std::cout << "logl after move: " << moved_logl << "\n";
            std::cout << "undo " << toString(candidates[j]) << "\n";
            undoMove(ann_network, candidates[j]);
            std::string debugInfoAfterUndo = exportDebugInfo(network);

            //std::cout << "network after undo move:\n";
            //std::cout << debugInfoAfterUndo << "\n";

            ASSERT_EQ(initialDebugInfo, debugInfoAfterUndo);
            double back_logl = computeLoglikelihood(ann_network);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void randomArcRemovalMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    Network &network = ann_network.network;
    std::string initialDebugInfo = exportDebugInfo(network);
    std::cout << initialDebugInfo;

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    std::vector<ArcRemovalMove> candidates = possibleArcRemovalMoves(ann_network);
    for (size_t j = 0; j < candidates.size(); ++j) {
        std::cout << "perform " << toString(candidates[j]);
        performMove(ann_network, candidates[j]);
        double moved_logl = computeLoglikelihood(ann_network);
        ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
        std::cout << "logl after move: " << moved_logl << "\n";
        std::cout << "undo " << toString(candidates[j]) << "\n";
        undoMove(ann_network, candidates[j]);
        std::string debugInfoAfterUndo = exportDebugInfo(network);
        EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
        double back_logl = computeLoglikelihood(ann_network);
        ASSERT_DOUBLE_EQ(initial_logl, back_logl);
    }
}

TEST (MovesTest, arcInsertionCeline) {
    randomArcInsertionMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, arcInsertionSmall) {
    randomArcInsertionMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, arcRemovalSmall) {
    randomArcRemovalMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, arcRemovalCeline) {
    randomArcRemovalMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, rnniSmall) {
    randomNNIMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, rnniCeline) {
    randomNNIMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, rspr1Small) {
    randomSPR1Moves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, rspr1Celine) {
    randomSPR1Moves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, rsprSmall) {
    randomSPRMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, rsprCeline) {
    randomSPRMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}
