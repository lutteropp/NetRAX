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
        std::vector<RNNIMove> candidates = possibleRNNIMoves(ann_network, network.edges_by_index[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {
            std::string newickBeforeMove = toExtendedNewick(network);
            std::cout << "perform " << toString(candidates[j]);
            performMove(ann_network, candidates[j]);
            std::cout << toExtendedNewick(network) << "\n";
            double moved_logl = computeLoglikelihood(ann_network);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            std::cout << "logl after move: " << moved_logl << "\n";
            std::cout << "undo " << toString(candidates[j]) << "\n";
            undoMove(ann_network, candidates[j]);
            std::string newickAfterUndoMove = toExtendedNewick(network);
            std::cout << toExtendedNewick(network) << "\n";
            std::string debugInfoAfterUndo = exportDebugInfo(network);
            EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
            double back_logl = computeLoglikelihood(ann_network);
            ASSERT_EQ(newickBeforeMove, newickAfterUndoMove);
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
        std::vector<RSPRMove> candidates = possibleRSPRMoves(ann_network, network.edges_by_index[i]);
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

void randomHeadMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
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
        std::vector<RSPRMove> candidates = possibleHeadMoves(ann_network, network.edges_by_index[i]);
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

void randomTailMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
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
        std::vector<RSPRMove> candidates = possibleTailMoves(ann_network, network.edges_by_index[i]);
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
        std::vector<RSPRMove> candidates = possibleRSPR1Moves(ann_network, network.edges_by_index[i]);
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
    //std::cout << initialDebugInfo;

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    //std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<ArcInsertionMove> candidates = possibleArcInsertionMoves(ann_network, network.edges_by_index[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {

            if (candidates[j].a_clv_index != 6 || candidates[j].b_clv_index != 7 || candidates[j].c_clv_index != 4
                    || candidates[j].d_clv_index != 3) {
                //    continue;
            }

            std::cout << "perform " << toString(candidates[j]);
            performMove(ann_network, candidates[j]);

            //std::cout << "network after move:\n";
            //std::cout << exportDebugInfo(network) << "\n";

            double moved_logl = computeLoglikelihood(ann_network);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            std::cout << "logl after move: " << moved_logl << "\n";
            std::cout << "undo " << toString(candidates[j]) << "\n";
            undoMove(ann_network, candidates[j]);
            //std::string debugInfoAfterUndo = exportDebugInfo(network);

            //std::cout << "network after undo move:\n";
            //std::cout << debugInfoAfterUndo << "\n";

            // ASSERT_EQ(initialDebugInfo, debugInfoAfterUndo);
            double back_logl = computeLoglikelihood(ann_network);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void randomDeltaPlusMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    Network &network = ann_network.network;
    std::string initialDebugInfo = exportDebugInfo(network);
    //std::cout << initialDebugInfo;

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    //std::cout << "initial_logl: " << initial_logl << "\n";

    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<ArcInsertionMove> candidates = possibleDeltaPlusMoves(ann_network, network.edges_by_index[i]);
        for (size_t j = 0; j < candidates.size(); ++j) {

            if (candidates[j].a_clv_index != 6 || candidates[j].b_clv_index != 7 || candidates[j].c_clv_index != 4
                    || candidates[j].d_clv_index != 3) {
                //    continue;
            }

            std::cout << "perform " << toString(candidates[j]);
            performMove(ann_network, candidates[j]);

            //std::cout << "network after move:\n";
            //std::cout << exportDebugInfo(network) << "\n";

            double moved_logl = computeLoglikelihood(ann_network);
            ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
            std::cout << "logl after move: " << moved_logl << "\n";
            std::cout << "undo " << toString(candidates[j]) << "\n";
            undoMove(ann_network, candidates[j]);
            //std::string debugInfoAfterUndo = exportDebugInfo(network);

            //std::cout << "network after undo move:\n";
            //std::cout << debugInfoAfterUndo << "\n";

            // ASSERT_EQ(initialDebugInfo, debugInfoAfterUndo);
            double back_logl = computeLoglikelihood(ann_network);
            ASSERT_DOUBLE_EQ(initial_logl, back_logl);
        }
    }
}

void printBranchLengths(AnnotatedNetwork &ann_network) {
    Network &network = ann_network.network;
    std::cout << "branch lengths:\n";
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::cout << "  " << network.edges[i].link1->node_clv_index << " -> " << network.edges[i].link2->node_clv_index
                << " has branch length: " << network.edges[i].length << "\n";
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
    //std::cout << initialDebugInfo;
    //printBranchLengths(ann_network);

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    std::vector<ArcRemovalMove> candidates = possibleArcRemovalMoves(ann_network);
    for (size_t j = 0; j < candidates.size(); ++j) {
        std::cout << "perform " << toString(candidates[j]);
        performMove(ann_network, candidates[j]);
        //std::cout << "network after move:\n";
        //std::cout << exportDebugInfo(network);
        //printBranchLengths(ann_network);
        double moved_logl = computeLoglikelihood(ann_network);
        ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
        std::cout << "logl after move: " << moved_logl << "\n";
        std::cout << "undo " << toString(candidates[j]) << "\n";
        undoMove(ann_network, candidates[j]);

        //printBranchLengths(ann_network);
        //std::string debugInfoAfterUndo = exportDebugInfo(network);
        //std::cout << debugInfoAfterUndo;
        //EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
        double back_logl = computeLoglikelihood(ann_network);
        ASSERT_DOUBLE_EQ(initial_logl, back_logl);
    }
}

void randomDeltaMinusMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    Network &network = ann_network.network;
    std::string initialDebugInfo = exportDebugInfo(network);
    //std::cout << initialDebugInfo;
    //printBranchLengths(ann_network);

    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";

    std::vector<ArcRemovalMove> candidates = possibleDeltaMinusMoves(ann_network);
    for (size_t j = 0; j < candidates.size(); ++j) {
        std::cout << "perform " << toString(candidates[j]);
        performMove(ann_network, candidates[j]);
        //std::cout << "network after move:\n";
        //std::cout << exportDebugInfo(network);
        //printBranchLengths(ann_network);
        double moved_logl = computeLoglikelihood(ann_network);
        ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
        std::cout << "logl after move: " << moved_logl << "\n";
        std::cout << "undo " << toString(candidates[j]) << "\n";
        undoMove(ann_network, candidates[j]);

        //printBranchLengths(ann_network);
        //std::string debugInfoAfterUndo = exportDebugInfo(network);
        //std::cout << debugInfoAfterUndo;
        //EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
        double back_logl = computeLoglikelihood(ann_network);
        ASSERT_DOUBLE_EQ(initial_logl, back_logl);
    }
}

TEST (MovesTest, tailSmall) {
    randomTailMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, headSmall) {
    randomHeadMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, tailCeline) {
    randomTailMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, headCeline) {
    randomHeadMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, arcInsertionSmall) {
    randomArcInsertionMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, deltaPlusSmall) {
    randomDeltaPlusMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, deltaMinusSmall) {
    randomDeltaMinusMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, deltaMinusCeline) {
    randomDeltaMinusMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, deltaPlusCeline) {
    randomDeltaPlusMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, rnniSmall) {
    randomNNIMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, rsprSmall) {
    randomSPRMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, rspr1Small) {
    randomSPR1Moves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, arcRemovalSmall) {
    randomArcRemovalMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST (MovesTest, rnniCeline) {
    randomNNIMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, rspr1Celine) {
    randomSPR1Moves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, rsprCeline) {
    randomSPRMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, arcRemovalCeline) {
    randomArcRemovalMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, arcInsertionCeline) {
    randomArcInsertionMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}
