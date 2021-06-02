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

#include "src/search/CandidateSelection.hpp"
#include "src/optimization/NetworkState.hpp"
#include "src/graph/NodeDisplayedTreeData.hpp"


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
    ASSERT_FLOAT_EQ(netrax::computeLoglikelihood(ann_network, 1, 1), netrax::computeLoglikelihood(ann_network, 0, 1));
    double initial_logl = computeLoglikelihood(ann_network);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());
    //std::cout << "initial_logl: " << initial_logl << "\n";
    std::string initialDebugInfo = exportDebugInfo(ann_network);
    //std::cout << initialDebugInfo << "\n";
    std::vector<std::vector<double> > old_brlens = extract_brlens(ann_network);
    NetworkState old_state = extract_network_state(ann_network);

    size_t old_num_active_trees = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees;

    std::cout << "Displayed trees at root before move:\n";
    for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++i) {
        DisplayedTreeData& dtd = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[i];
        assert(!dtd.treeLoglData.reticulationChoices.empty());
        std::cout << "tree " << i << "logprob: " << dtd.treeLoglData.tree_logprob << "\n";
        printReticulationChoices(dtd.treeLoglData.reticulationChoices);
    }

    for (size_t j = 0; j < candidates.size(); ++j) {
        std::cout << "Testing moves for candidate " << j+1 << "/" << candidates.size() << "...\n";
        std::string newickBeforeMove = toExtendedNewick(ann_network);

        Move origMove(candidates[j]);

        //std::cout << "perform " << toString(candidates[j]);
        performMove(ann_network, candidates[j]);

        ASSERT_DOUBLE_EQ(netrax::computeLoglikelihood(ann_network, 1, 1), netrax::computeLoglikelihood(ann_network, 0, 1));

        //std::cout << toExtendedNewick(network) << "\n";
        double moved_logl = computeLoglikelihood(ann_network);
        ASSERT_NE(moved_logl, -std::numeric_limits<double>::infinity());
        //std::cout << "logl after move: " << moved_logl << "\n";
        //std::cout << "undo " << toString(candidates[j]) << "\n";
        undoMove(ann_network, candidates[j]);

        NetworkState new_state = extract_network_state(ann_network);
        ASSERT_TRUE(network_states_equal(old_state, new_state));

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
        std::string debugInfoAfterUndo = exportDebugInfo(ann_network);

        if (initialDebugInfo != debugInfoAfterUndo) {
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << "inequal debug infos:\n\n";
                std::cout << initialDebugInfo << "\n\n";
                std::cout << debugInfoAfterUndo << "\n";
            }
        }

        std::cout << "Displayed trees at root after undo move:\n";
        for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++i) {
            DisplayedTreeData& dtd = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[i];
            assert(!dtd.treeLoglData.reticulationChoices.empty());
            std::cout << "tree " << i << " logprob: " << dtd.treeLoglData.tree_logprob << "\n";
            printReticulationChoices(dtd.treeLoglData.reticulationChoices);
        }

        EXPECT_EQ(initialDebugInfo, debugInfoAfterUndo);
        ASSERT_DOUBLE_EQ(netrax::computeLoglikelihood(ann_network, 1, 1), netrax::computeLoglikelihood(ann_network, 0, 1));
        double back_logl = computeLoglikelihood(ann_network);
        ASSERT_GE(ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees, 1);

        size_t act_num_active_trees = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees;
        ASSERT_EQ(old_num_active_trees, act_num_active_trees);

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

void twoMovesStep(AnnotatedNetwork& ann_network, MoveType type) {
    ASSERT_DOUBLE_EQ(netrax::computeLoglikelihood(ann_network, 1, 1), netrax::computeLoglikelihood(ann_network, 0, 1));
    std::vector<Move> candidates = possibleMoves(ann_network, type);
    if (!candidates.empty()) {
        performMove(ann_network, candidates[0]);
        ASSERT_DOUBLE_EQ(netrax::computeLoglikelihood(ann_network, 1, 1), netrax::computeLoglikelihood(ann_network, 0, 1));

        updateOldCandidates(ann_network, candidates[0], candidates);
        removeBadCandidates(ann_network, candidates);

        if (candidates.empty()) {
            candidates = possibleMoves(ann_network, type);
        }

        if (!candidates.empty()) {
            performMove(ann_network, candidates[0]);
            ASSERT_DOUBLE_EQ(netrax::computeLoglikelihood(ann_network, 1, 1), netrax::computeLoglikelihood(ann_network, 0, 1));
        }
    }
}

void twoMoves(const std::string &networkPath, const std::string &msaPath, bool useRepeats,
        MoveType type) {
    NetraxOptions options;
    options.run_single_threaded = true;
    options.start_network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    const RaxmlInstance instance = createRaxmlInstance(options);
    AnnotatedNetwork ann_network = build_annotated_network(options, instance);
    init_annotated_network(ann_network);

    twoMovesStep(ann_network, type);
}

void insertAndRemove(const std::string &networkPath, const std::string& msaPath, bool useRepeats) {
    NetraxOptions options;
    options.run_single_threaded = true;
    options.start_network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    const RaxmlInstance instance = createRaxmlInstance(options);
    AnnotatedNetwork ann_network = build_annotated_network(options, instance);
    init_annotated_network(ann_network);

    std::vector<Move> insertionCandidates = possibleMoves(ann_network, MoveType::ArcInsertionMove);
    for (size_t i = 0; i < 10; ++i) {
        performMove(ann_network, insertionCandidates[0]);
        std::vector<Move> removalCandidates = possibleMoves(ann_network, MoveType::ArcRemovalMove);
        performMove(ann_network, removalCandidates[0]);
        updateOldCandidates(ann_network, removalCandidates[0], insertionCandidates);
        removeBadCandidates(ann_network, insertionCandidates);
        if (insertionCandidates.empty()) {
            insertionCandidates = possibleMoves(ann_network, MoveType::ArcInsertionMove);
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

TEST (MovesTest, interleavedRemoval) {
    insertAndRemove(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
    insertAndRemove(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST (MovesTest, tail) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::TailMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::TailMove);
}

TEST (MovesTest, twomoves_tail) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::TailMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::TailMove);
}

TEST (MovesTest, head) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::HeadMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::HeadMove);
}

TEST (MovesTest, twomoves_head) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::HeadMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::HeadMove);
}

TEST (MovesTest, arcInsertion) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::ArcInsertionMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::ArcInsertionMove);
}

TEST (MovesTest, twomoves_arcInsertion) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::ArcInsertionMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::ArcInsertionMove);
}

TEST (MovesTest, deltaPlus) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::DeltaPlusMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::DeltaPlusMove);
}

TEST (MovesTest, twomoves_deltaPlus) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::DeltaPlusMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::DeltaPlusMove);
}

TEST (MovesTest, arcRemoval) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::ArcRemovalMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::ArcRemovalMove);
}

TEST (MovesTest, twomoves_arcRemoval) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::ArcRemovalMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::ArcRemovalMove);
}

TEST (MovesTest, deltaMinus) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::DeltaMinusMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::DeltaMinusMove);
}

TEST (MovesTest, twomoves_deltaMinus) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::DeltaMinusMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::DeltaMinusMove);
}

TEST (MovesTest, rnni) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RNNIMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RNNIMove);
}

TEST (MovesTest, twomoves_rnni) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RNNIMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RNNIMove);
}

TEST (MovesTest, rspr) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RSPRMove);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RSPRMove);
}

TEST (MovesTest, twomoves_rspr) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RSPRMove);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RSPRMove);
}

TEST (MovesTest, rspr1) {
    randomMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RSPR1Move);
    randomMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RSPR1Move);
}

TEST (MovesTest, twomoves_rspr1) {
    twoMoves(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false,
            MoveType::RSPR1Move);
    twoMoves(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false,
            MoveType::RSPR1Move);
}
