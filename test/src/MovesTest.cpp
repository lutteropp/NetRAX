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
#include <algorithm>

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "sample_networks/";

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
        std::cout << "Testing moves for candidate " << j+1 << "/" << candidates.size() << "...\n";
        std::string newickBeforeMove = toExtendedNewick(ann_network);

        Move origMove(candidates[j]);

        //std::cout << "perform " << toString(candidates[j]);
        performMove(ann_network, candidates[j]);
        //std::cout << toExtendedNewick(network) << "\n";
        double moved_logl = computeLoglikelihood(ann_network);
        ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
        //std::cout << "logl after move: " << moved_logl << "\n";
        //std::cout << "undo " << toString(candidates[j]) << "\n";
        undoMove(ann_network, candidates[j]);

        ASSERT_EQ(origMove.arcRemovalData.a_clv_index, candidates[j].arcRemovalData.a_clv_index);
        ASSERT_EQ(origMove.arcRemovalData.b_clv_index, candidates[j].arcRemovalData.b_clv_index);
        ASSERT_EQ(origMove.arcRemovalData.c_clv_index, candidates[j].arcRemovalData.c_clv_index);
        ASSERT_EQ(origMove.arcRemovalData.d_clv_index, candidates[j].arcRemovalData.d_clv_index);
        ASSERT_EQ(origMove.arcRemovalData.u_clv_index, candidates[j].arcRemovalData.u_clv_index);
        ASSERT_EQ(origMove.arcRemovalData.v_clv_index, candidates[j].arcRemovalData.v_clv_index);
        ASSERT_EQ(origMove.arcRemovalData.au_pmatrix_index, candidates[j].arcRemovalData.au_pmatrix_index);
        ASSERT_EQ(origMove.arcRemovalData.cv_pmatrix_index, candidates[j].arcRemovalData.cv_pmatrix_index);
        ASSERT_EQ(origMove.arcRemovalData.ub_pmatrix_index, candidates[j].arcRemovalData.ub_pmatrix_index);
        ASSERT_EQ(origMove.arcRemovalData.uv_pmatrix_index, candidates[j].arcRemovalData.uv_pmatrix_index);
        ASSERT_EQ(origMove.arcRemovalData.vd_pmatrix_index, candidates[j].arcRemovalData.vd_pmatrix_index);

        ASSERT_EQ(origMove.arcInsertionData.a_clv_index, candidates[j].arcInsertionData.a_clv_index);
        ASSERT_EQ(origMove.arcInsertionData.b_clv_index, candidates[j].arcInsertionData.b_clv_index);
        ASSERT_EQ(origMove.arcInsertionData.c_clv_index, candidates[j].arcInsertionData.c_clv_index);
        ASSERT_EQ(origMove.arcInsertionData.d_clv_index, candidates[j].arcInsertionData.d_clv_index);
        ASSERT_EQ(origMove.arcInsertionData.ab_pmatrix_index, candidates[j].arcInsertionData.ab_pmatrix_index);
        ASSERT_EQ(origMove.arcInsertionData.cd_pmatrix_index, candidates[j].arcInsertionData.cd_pmatrix_index);

        ASSERT_EQ(origMove.rsprData.x_clv_index, candidates[j].rsprData.x_clv_index);
        ASSERT_EQ(origMove.rsprData.y_clv_index, candidates[j].rsprData.y_clv_index);
        ASSERT_EQ(origMove.rsprData.z_clv_index, candidates[j].rsprData.z_clv_index);
        ASSERT_EQ(origMove.rsprData.x_prime_clv_index, candidates[j].rsprData.x_prime_clv_index);
        ASSERT_EQ(origMove.rsprData.y_prime_clv_index, candidates[j].rsprData.y_prime_clv_index);

        ASSERT_EQ(origMove.rnniData.u_clv_index, candidates[j].rnniData.u_clv_index);
        ASSERT_EQ(origMove.rnniData.v_clv_index, candidates[j].rnniData.v_clv_index);
        ASSERT_EQ(origMove.rnniData.s_clv_index, candidates[j].rnniData.s_clv_index);
        ASSERT_EQ(origMove.rnniData.t_clv_index, candidates[j].rnniData.t_clv_index);

        std::vector<std::vector<double> > act_brlens = extract_brlens(ann_network);
        for (size_t i = 0; i < act_brlens.size(); ++i) {
            for (size_t j = 0; j < act_brlens[i].size(); ++j) {
                if (act_brlens[i][j] != old_brlens[i][j]) {
                    std::cout << "problem at pmatrix index " << j << "\n";
                }
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

    std::vector<Move> candidates = possibleMoves(ann_network, type);

    size_t max_candidates = candidates.size();// 200;

    std::random_shuffle(candidates.begin(), candidates.end());
    candidates.resize(std::min(candidates.size(), max_candidates));

    randomMovesStep(ann_network, candidates);
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
