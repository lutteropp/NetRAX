/*
 * MovesTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/optimization/Moves.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"
#include "src/DebugPrintFunctions.hpp"

#include "src/graph/NetworkFunctions.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "examples/sample_networks/";

std::vector<std::vector<double> > extract_brlens(AnnotatedNetwork &ann_network) {
    std::vector<std::vector<double> > res;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t n_partitions = 1;
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    res.resize(n_partitions);
    for (size_t p = 0; p < n_partitions; ++p) {
        res[p].resize(ann_network.network.edges.size());
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            res[p][pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    }
    return res;
}

template<typename T>
void randomMovesStep(AnnotatedNetwork &ann_network, std::vector<T> candidates) {
    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    std::cout << "initial_logl: " << initial_logl << "\n";
    //std::string initialDebugInfo = exportDebugInfo(network);
    //std::cout << initialDebugInfo << "\n";
    std::vector<std::vector<double> > old_brlens = extract_brlens(ann_network);

    for (size_t j = 0; j < candidates.size(); ++j) {
        std::string newickBeforeMove = toExtendedNewick(ann_network);
        //std::cout << "perform " << toString(candidates[j]);
        performMove(ann_network, candidates[j]);
        //std::cout << toExtendedNewick(network) << "\n";
        double moved_logl = computeLoglikelihood(ann_network);
        ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
        //std::cout << "logl after move: " << moved_logl << "\n";
        //std::cout << "undo " << toString(candidates[j]) << "\n";
        undoMove(ann_network, candidates[j]);
        computeLoglikelihood(ann_network);
        std::vector<std::vector<double> > act_brlens = extract_brlens(ann_network);
        for (size_t i = 0; i < act_brlens.size(); ++i) {
            for (size_t j = 0; j < act_brlens[i].size(); ++j) {
                ASSERT_DOUBLE_EQ(act_brlens[i][j], old_brlens[i][j]);
            }
        }
        //std::string newickAfterUndoMove = toExtendedNewick(network);
        //std::cout << toExtendedNewick(network) << "\n";
        //std::string debugInfoAfterUndo = exportDebugInfo(network);
        //EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
        double back_logl = computeLoglikelihood(ann_network);
        //ASSERT_EQ(newickBeforeMove, newickAfterUndoMove);
        ASSERT_DOUBLE_EQ(initial_logl, back_logl);
    }
}

void randomMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats,
        MoveType type) {
    NetraxOptions options;
    options.start_network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = build_annotated_network(options);
    init_annotated_network(ann_network);
    Network &network = ann_network.network;

    if (type == MoveType::ArcRemovalMove) {
        randomMovesStep<ArcRemovalMove>(ann_network, possibleArcRemovalMoves(ann_network));
        return;
    } else if (type == MoveType::DeltaMinusMove) {
        randomMovesStep<ArcRemovalMove>(ann_network, possibleDeltaMinusMoves(ann_network));
        return;
    }

    for (size_t i = 0; i < network.num_branches(); ++i) {
        switch (type) {
        case MoveType::RNNIMove:
            randomMovesStep<RNNIMove>(ann_network,
                    possibleRNNIMoves(ann_network, &network.edges[i]));
            break;
        case MoveType::RSPRMove:
            randomMovesStep<RSPRMove>(ann_network,
                    possibleRSPRMoves(ann_network, &network.edges[i]));
            break;
        case MoveType::RSPR1Move:
            randomMovesStep<RSPRMove>(ann_network,
                    possibleRSPR1Moves(ann_network, &network.edges[i]));
            break;
        case MoveType::HeadMove:
            randomMovesStep<RSPRMove>(ann_network,
                    possibleHeadMoves(ann_network, &network.edges[i]));
            break;
        case MoveType::TailMove:
            randomMovesStep<RSPRMove>(ann_network,
                    possibleTailMoves(ann_network, &network.edges[i]));
            break;
        case MoveType::ArcInsertionMove:
            randomMovesStep<ArcInsertionMove>(ann_network,
                    possibleArcInsertionMoves(ann_network, &network.edges[i]));
            break;
        case MoveType::DeltaPlusMove:
            randomMovesStep<ArcInsertionMove>(ann_network,
                    possibleDeltaPlusMoves(ann_network, &network.edges[i]));
            break;
        default:
            throw std::runtime_error("Invalid move type");
        }
    }
}

void printBranchLengths(AnnotatedNetwork &ann_network) {
    Network &network = ann_network.network;
    std::cout << "branch lengths:\n";
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::cout << "  " << network.edges[i].link1->node_clv_index << " -> "
                << network.edges[i].link2->node_clv_index << " has branch length: "
                << network.edges[i].length << "\n";
    }
}

TEST (MovesTest, tail) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::TailMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::TailMove);
}

TEST (MovesTest, head) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::HeadMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::HeadMove);
}

TEST (MovesTest, arcInsertion) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::ArcInsertionMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::ArcInsertionMove);
}

TEST (MovesTest, deltaPlus) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::DeltaPlusMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::DeltaPlusMove);
}

TEST (MovesTest, arcRemoval) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::ArcRemovalMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::ArcRemovalMove);
}

TEST (MovesTest, deltaMinus) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::DeltaMinusMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::DeltaMinusMove);
}

TEST (MovesTest, rnni) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RNNIMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RNNIMove);
}

TEST (MovesTest, rspr) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RSPRMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RSPRMove);
}

TEST (MovesTest, rspr1) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RSPR1Move);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RSPR1Move);
}
