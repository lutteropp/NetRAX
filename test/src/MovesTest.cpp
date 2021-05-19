/*
 * MovesTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/moves/Move.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"
#include "src/DebugPrintFunctions.hpp"

#include "src/helper/NetworkFunctions.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "/home/sarah/code-workspace/NetRAX/test/sample_networks/";

std::vector<std::vector<double> > extract_brlens(AnnotatedNetwork &ann_network) {
    std::vector<std::vector<double> > res;
    if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        res.resize(ann_network.fake_treeinfo->partition_count);
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote branch lengths
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            res[p].resize(ann_network.network.num_branches());
            for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
                res[p][i] = ann_network.fake_treeinfo->branch_lengths[p][i];
            }
        }
    } else {
        res.resize(1);
        res[0].resize(ann_network.network.num_branches());
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            res[0][i] = ann_network.fake_treeinfo->linked_branch_lengths[i];
        }
    }
    return res;
}

void randomMovesStep(AnnotatedNetwork &ann_network, std::vector<Move> candidates) {
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
    options.run_single_threaded = true;
    options.start_network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    const RaxmlInstance instance = createRaxmlInstance(options);
    AnnotatedNetwork ann_network = build_annotated_network(options, instance);
    init_annotated_network(ann_network);
    Network &network = ann_network.network;

    if (type == MoveType::ArcRemovalMove || type == MoveType::DeltaMinusMove) {
        randomMovesStep(ann_network, possibleMoves(ann_network, type));
        return;
    }

    for (size_t i = 0; i < network.num_branches(); ++i) {
        randomMovesStep(ann_network, possibleMoves(ann_network, type, ann_network.network.edges_by_index[i]));
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
