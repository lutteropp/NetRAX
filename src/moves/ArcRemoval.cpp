#include "ArcRemoval.hpp"
#include "ArcInsertion.hpp"

#include <utility>
#include <numeric>
#include <algorithm>

#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"

#include "../DebugPrintFunctions.hpp"
#include "GeneralMoveFunctions.hpp"

namespace netrax {

bool checkSanityArcRemoval(AnnotatedNetwork& ann_network, const Move& move) {
    bool good = true;
    good &= (move.moveType == MoveType::ArcRemovalMove || move.moveType == MoveType::DeltaMinusMove);
    good &= (ann_network.network.nodes_by_index[move.arcRemovalData.a_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.arcRemovalData.b_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.arcRemovalData.c_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.arcRemovalData.d_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.arcRemovalData.u_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.arcRemovalData.v_clv_index] != nullptr);
    good &= (ann_network.network.edges_by_index[move.arcRemovalData.au_pmatrix_index] != nullptr);
    good &= (ann_network.network.edges_by_index[move.arcRemovalData.ub_pmatrix_index] != nullptr);
    good &= (ann_network.network.edges_by_index[move.arcRemovalData.uv_pmatrix_index] != nullptr);
    good &= (ann_network.network.edges_by_index[move.arcRemovalData.cv_pmatrix_index] != nullptr);
    good &= (ann_network.network.edges_by_index[move.arcRemovalData.vd_pmatrix_index] != nullptr);
    good &= (move.arcRemovalData.a_clv_index != move.arcRemovalData.u_clv_index);
    good &= (move.arcRemovalData.u_clv_index != move.arcRemovalData.b_clv_index);
    good &= (move.arcRemovalData.c_clv_index != move.arcRemovalData.v_clv_index);
    good &= (move.arcRemovalData.v_clv_index != move.arcRemovalData.d_clv_index);
    good &= (move.arcRemovalData.u_clv_index != move.arcRemovalData.v_clv_index);
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.arcRemovalData.a_clv_index], ann_network.network.nodes_by_index[move.arcRemovalData.u_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.arcRemovalData.u_clv_index], ann_network.network.nodes_by_index[move.arcRemovalData.b_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.arcRemovalData.u_clv_index], ann_network.network.nodes_by_index[move.arcRemovalData.v_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.arcRemovalData.v_clv_index], ann_network.network.nodes_by_index[move.arcRemovalData.d_clv_index]));
    return good;
}

bool checkSanityArcRemoval(AnnotatedNetwork& ann_network, std::vector<Move>& moves) {
    bool sane = true;
    for (size_t i = 0; i < moves.size(); ++i) {
        sane &= checkSanityArcRemoval(ann_network, moves[i]);
    }
    return sane;
}

void updateMoveClvIndex(Move& move, size_t old_clv_index, size_t new_clv_index) {
    Move origMove = move;
    // set new to old
    if (origMove.arcRemovalData.a_clv_index == new_clv_index) {
        move.arcRemovalData.a_clv_index = old_clv_index;
    }
    if (origMove.arcRemovalData.b_clv_index == new_clv_index) {
        move.arcRemovalData.b_clv_index = old_clv_index;
    }
    if (origMove.arcRemovalData.c_clv_index == new_clv_index) {
        move.arcRemovalData.c_clv_index = old_clv_index;
    }
    if (origMove.arcRemovalData.d_clv_index == new_clv_index) {
        move.arcRemovalData.d_clv_index = old_clv_index;
    }
    if (origMove.arcRemovalData.u_clv_index == new_clv_index) {
        move.arcRemovalData.u_clv_index = old_clv_index;
    }
    if (origMove.arcRemovalData.v_clv_index == new_clv_index) {
        move.arcRemovalData.v_clv_index = old_clv_index;
    }

    // set old to new
    if (origMove.arcRemovalData.a_clv_index == old_clv_index) {
        move.arcRemovalData.a_clv_index = new_clv_index;    
    }
    if (origMove.arcRemovalData.b_clv_index == old_clv_index) {
        move.arcRemovalData.b_clv_index = new_clv_index;
    }
    if (origMove.arcRemovalData.c_clv_index == old_clv_index) {
        move.arcRemovalData.c_clv_index = new_clv_index;
    }
    if (origMove.arcRemovalData.d_clv_index == old_clv_index) {
        move.arcRemovalData.d_clv_index = new_clv_index;
    }
    if (origMove.arcRemovalData.u_clv_index == old_clv_index) {
        move.arcRemovalData.u_clv_index = new_clv_index;
    }
    if (origMove.arcRemovalData.v_clv_index == old_clv_index) {
        move.arcRemovalData.v_clv_index = new_clv_index;
    }
}

void updateMovePmatrixIndex(Move& move, size_t old_pmatrix_index, size_t new_pmatrix_index) {
    Move origMove = move;
    // set new to old
    if (origMove.arcRemovalData.au_pmatrix_index == new_pmatrix_index) {
        move.arcRemovalData.au_pmatrix_index = old_pmatrix_index;
    }
    if (origMove.arcRemovalData.cv_pmatrix_index == new_pmatrix_index) {
        move.arcRemovalData.cv_pmatrix_index = old_pmatrix_index;
    }
    if (origMove.arcRemovalData.ub_pmatrix_index == new_pmatrix_index) {
        move.arcRemovalData.ub_pmatrix_index = old_pmatrix_index;
    }
    if (origMove.arcRemovalData.uv_pmatrix_index == new_pmatrix_index) {
        move.arcRemovalData.uv_pmatrix_index = old_pmatrix_index;
    }
    if (origMove.arcRemovalData.vd_pmatrix_index == new_pmatrix_index) {
        move.arcRemovalData.vd_pmatrix_index = old_pmatrix_index;
    }

    // set old to new
    if (origMove.arcRemovalData.au_pmatrix_index == old_pmatrix_index) {
        move.arcRemovalData.au_pmatrix_index = new_pmatrix_index;    
    }
    if (origMove.arcRemovalData.cv_pmatrix_index == old_pmatrix_index) {
        move.arcRemovalData.cv_pmatrix_index = new_pmatrix_index;
    }
    if (origMove.arcRemovalData.ub_pmatrix_index == old_pmatrix_index) {
        move.arcRemovalData.ub_pmatrix_index = new_pmatrix_index;
    }
    if (origMove.arcRemovalData.uv_pmatrix_index == old_pmatrix_index) {
        move.arcRemovalData.uv_pmatrix_index = new_pmatrix_index;
    }
    if (origMove.arcRemovalData.vd_pmatrix_index == old_pmatrix_index) {
        move.arcRemovalData.vd_pmatrix_index = new_pmatrix_index;
    }
}

void fixReticulationsArcRemoval(Network &network, Move &move) {
    Move remapped_move(move);
    for (size_t i = 0; i < move.arcRemovalData.remapped_clv_indices.size(); ++i) {
        size_t old_clv_index = move.arcRemovalData.remapped_clv_indices[i].first;
        size_t new_clv_index = move.arcRemovalData.remapped_clv_indices[i].second;
        updateMoveClvIndex(remapped_move, old_clv_index, new_clv_index);
    }

    std::unordered_set<Node*> repair_candidates;
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[remapped_move.arcRemovalData.a_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[remapped_move.arcRemovalData.b_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[remapped_move.arcRemovalData.c_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[remapped_move.arcRemovalData.d_clv_index]);
    for (Node *node : repair_candidates) {
        if (node->type == NodeType::RETICULATION_NODE) {
            resetReticulationLinks(node);
        }
    }
}

Move buildMoveArcRemoval(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, size_t u_clv_index, size_t v_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
         std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType, size_t edge_orig_idx, size_t node_orig_idx) {
    Move move = Move(moveType, edge_orig_idx, node_orig_idx);
    move.arcRemovalData.a_clv_index = a_clv_index;
    move.arcRemovalData.b_clv_index = b_clv_index;
    move.arcRemovalData.c_clv_index = c_clv_index;
    move.arcRemovalData.d_clv_index = d_clv_index;
    move.arcRemovalData.u_clv_index = u_clv_index;
    move.arcRemovalData.v_clv_index = v_clv_index;

    move.arcRemovalData.u_v_len = u_v_len;
    move.arcRemovalData.c_v_len = c_v_len;
    move.arcRemovalData.a_u_len = a_u_len;
    
    move.arcRemovalData.a_b_len = a_b_len;
    move.arcRemovalData.c_d_len = c_d_len;
    move.arcRemovalData.v_d_len = v_d_len;
    move.arcRemovalData.u_b_len = u_b_len;
    move.moveType = moveType;
    return move;
}

std::vector<Move> possibleMovesArcRemoval(AnnotatedNetwork &ann_network, Node *v,
        size_t edge_orig_idx, MoveType moveType) {
// v is a reticulation node, u is one parent of v, c is the other parent of v, a is parent of u, d is child of v, b is other child of u
    std::vector<Move> res;
    Network &network = ann_network.network;
    assert(v->type == NodeType::RETICULATION_NODE);
    Node *d = getReticulationChild(network, v);
    Node *first_parent = getReticulationFirstParent(network, v);
    Node *second_parent = getReticulationSecondParent(network, v);
    std::vector<std::pair<Node*, Node*> > ucChoices;
    double max_br = ann_network.options.brlen_max;
    if (first_parent != network.root && first_parent->type != NodeType::RETICULATION_NODE) {
        ucChoices.emplace_back(std::make_pair(first_parent, second_parent));
    }
    if (second_parent != network.root && second_parent->type != NodeType::RETICULATION_NODE) {
        ucChoices.emplace_back(std::make_pair(second_parent, first_parent));
    }
    for (size_t i = 0; i < ucChoices.size(); ++i) {
        Node *u = ucChoices[i].first;
        Node *c = ucChoices[i].second;
        Node *b = getOtherChild(network, u, v);
        if (hasChild(network, c, d)) { // avoid creating parallel arcs
            continue;
        }
        assert(u);
        assert(c);
        assert(b);
        Node *a = getActiveParent(network, u);
        assert(a);
        if (hasChild(network, a, b)) { // avoid creating parallel arcs
            continue;
        }

        if (a->clv_index == c->clv_index && b->clv_index == d->clv_index) {
            continue;
        }

        std::vector<double> a_u_len, u_b_len, a_b_len, c_v_len, v_d_len, c_d_len, u_v_len;
        a_u_len = get_edge_lengths(ann_network, getEdgeTo(network, a, u)->pmatrix_index);
        u_b_len = get_edge_lengths(ann_network, getEdgeTo(network, u, b)->pmatrix_index);
        a_b_len = get_plus_edge_lengths(a_u_len, u_b_len, max_br);
        c_v_len = get_edge_lengths(ann_network, getEdgeTo(network, c, v)->pmatrix_index);
        v_d_len = get_edge_lengths(ann_network, getEdgeTo(network, v, d)->pmatrix_index);
        c_d_len = get_plus_edge_lengths(c_v_len, v_d_len, max_br);
        u_v_len = get_edge_lengths(ann_network, getEdgeTo(network, u, v)->pmatrix_index);

        Move move = buildMoveArcRemoval(a->clv_index, b->clv_index, c->clv_index,
                d->clv_index, u->clv_index, v->clv_index, u_v_len,
                c_v_len, a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType, edge_orig_idx, v->clv_index);

        move.arcRemovalData.au_pmatrix_index = getEdgeTo(network, a, u)->pmatrix_index;
        move.arcRemovalData.ub_pmatrix_index = getEdgeTo(network, u, b)->pmatrix_index;
        move.arcRemovalData.cv_pmatrix_index = getEdgeTo(network, c, v)->pmatrix_index;
        move.arcRemovalData.vd_pmatrix_index = getEdgeTo(network, v, d)->pmatrix_index;
        move.arcRemovalData.uv_pmatrix_index = getEdgeTo(network, u, v)->pmatrix_index;

        assert(move.arcRemovalData.a_clv_index != move.arcRemovalData.u_clv_index);

        assert(checkSanityArcRemoval(ann_network, move));

        res.emplace_back(move);
    }
    return res;
}

std::vector<Move> possibleMovesDeltaMinus(AnnotatedNetwork &ann_network, Node *v, size_t edge_orig_idx) {
    std::vector<Move> res;
    std::vector<Move> allRemovals = possibleMovesArcRemoval(ann_network, v, edge_orig_idx,
            MoveType::DeltaMinusMove);
// 3 cases: a == c, b == d, or b == c
    for (size_t i = 0; i < allRemovals.size(); ++i) {
        if ((allRemovals[i].arcRemovalData.a_clv_index == allRemovals[i].arcRemovalData.c_clv_index)
                || (allRemovals[i].arcRemovalData.b_clv_index == allRemovals[i].arcRemovalData.d_clv_index)
                || (allRemovals[i].arcRemovalData.b_clv_index == allRemovals[i].arcRemovalData.c_clv_index)) {
            res.emplace_back(allRemovals[i]);
        }
    }
    return res;
}

std::vector<Move> possibleMovesArcRemoval(AnnotatedNetwork &ann_network) {
    std::vector<Move> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        size_t edge_orig_idx = getReticulationFirstParentPmatrixIndex(ann_network.network.reticulation_nodes[i]);
        auto moves = possibleMovesArcRemoval(ann_network, network.reticulation_nodes[i], edge_orig_idx,
                MoveType::ArcRemovalMove);
        res.insert(std::end(res), std::begin(moves), std::end(moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanityArcRemoval(ann_network, res));
    return res;
}

std::vector<Move> possibleMovesDeltaMinus(AnnotatedNetwork &ann_network) {
    std::vector<Move> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        size_t edge_orig_idx = getReticulationFirstParentPmatrixIndex(ann_network.network.reticulation_nodes[i]);
        std::vector<Move> branch_moves = possibleMovesDeltaMinus(ann_network,
                network.reticulation_nodes[i], edge_orig_idx);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanityArcRemoval(ann_network, res));
    return res;
}

std::vector<Move> possibleMovesArcRemoval(AnnotatedNetwork& ann_network, const std::vector<Node*>& start_nodes) {
    std::vector<Move> res;
    for (Node* node : start_nodes) {
        if (node->getType() == NodeType::RETICULATION_NODE) {
            size_t edge_orig_idx = getReticulationFirstParentPmatrixIndex(node);
            auto moves = possibleMovesArcRemoval(ann_network, node, edge_orig_idx,
                    MoveType::ArcRemovalMove);
            res.insert(std::end(res), std::begin(moves), std::end(moves));
        }
    }
    sortByProximity(res, ann_network);
    assert(checkSanityArcRemoval(ann_network, res));
    return res;
}

std::vector<Move> possibleMovesDeltaMinus(AnnotatedNetwork& ann_network, const std::vector<Node*>& start_nodes) {
    std::vector<Move> res;
    for (Node* node : start_nodes) {
        if (node->getType() == NodeType::RETICULATION_NODE) {
            size_t edge_orig_idx = getReticulationFirstParentPmatrixIndex(node);
            auto moves = possibleMovesDeltaMinus(ann_network, node, edge_orig_idx);
            res.insert(std::end(res), std::begin(moves), std::end(moves));
        }
    }
    sortByProximity(res, ann_network);
    assert(checkSanityArcRemoval(ann_network, res));
    return res;
}

std::vector<Move> possibleMovesArcRemoval(AnnotatedNetwork& ann_network, const std::vector<Edge*>& start_edges) {
    std::vector<Move> res;
    for (Edge* edge : start_edges) {
        Node* node = getTarget(ann_network.network, edge);
        if (node->getType() == NodeType::RETICULATION_NODE) {
            size_t edge_orig_idx = edge->pmatrix_index;
            auto moves = possibleMovesArcRemoval(ann_network, node, edge_orig_idx,
                    MoveType::ArcRemovalMove);
            res.insert(std::end(res), std::begin(moves), std::end(moves));
        }
    }
    sortByProximity(res, ann_network);
    assert(checkSanityArcRemoval(ann_network, res));
    return res;
}

std::vector<Move> possibleMovesDeltaMinus(AnnotatedNetwork& ann_network, const std::vector<Edge*>& start_edges) {
    std::vector<Move> res;
    for (Edge* edge : start_edges) {
        Node* node = getTarget(ann_network.network, edge);
        if (node->getType() == NodeType::RETICULATION_NODE) {
            size_t edge_orig_idx = edge->pmatrix_index;
            auto moves = possibleMovesDeltaMinus(ann_network, node, edge_orig_idx);
            res.insert(std::end(res), std::begin(moves), std::end(moves));
        }
    }
    sortByProximity(res, ann_network);
    assert(checkSanityArcRemoval(ann_network, res));
    return res;
}

void repairConsecutiveClvIndices(AnnotatedNetwork &ann_network, Move& move) {
    std::vector<size_t> missing_clv_indices;
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        if (!ann_network.network.nodes_by_index[i]) {
            missing_clv_indices.emplace_back(i);
            invalidateSingleClv(ann_network, i);
        }
    }

    if (missing_clv_indices.empty()) {
        return;
    }

    for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
        if (ann_network.network.nodes[i].clv_index >= ann_network.network.num_nodes() && ann_network.network.nodes[i].clv_index < std::numeric_limits<size_t>::max()) {
            size_t old_clv_index = ann_network.network.nodes[i].clv_index;
            size_t new_clv_index = missing_clv_indices.back();

            move.arcRemovalData.remapped_clv_indices.emplace_back(std::make_pair(old_clv_index, new_clv_index));

            // invalidate the clv entry
            invalidateSingleClv(ann_network, old_clv_index);

            // update all references to this clv index
            ann_network.network.nodes[i].clv_index = new_clv_index;
            ann_network.network.nodes[i].scaler_index = new_clv_index - ann_network.network.num_tips();
            ann_network.network.nodes_by_index[new_clv_index] = &ann_network.network.nodes[i];
            ann_network.network.nodes_by_index[old_clv_index] = nullptr;
            for (size_t j = 0; j < ann_network.network.nodes[i].links.size(); ++j) {
                ann_network.network.nodes[i].links[j].node_clv_index = new_clv_index;
            }

            missing_clv_indices.pop_back();
        }
    }
}

void repairConsecutivePmatrixIndices(AnnotatedNetwork &ann_network, Move& move) {
    std::vector<size_t> missing_pmatrix_indices;
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        if (!ann_network.network.edges_by_index[i]) {
            missing_pmatrix_indices.emplace_back(i);
            std::vector<bool> visited(ann_network.network.edges.size(), false);
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                // skip remote partitions
                if (!ann_network.fake_treeinfo->partitions[p]) {
                    continue;
                }
                ann_network.fake_treeinfo->pmatrix_valid[p][i] = 0;
            }
        }
    }

    if (missing_pmatrix_indices.empty()) {
        return;
    }

    for (size_t i = 0; i < ann_network.network.edges.size(); ++i) {
        if (ann_network.network.edges[i].pmatrix_index >= ann_network.network.num_branches() && ann_network.network.edges[i].pmatrix_index < std::numeric_limits<size_t>::max()) {
            size_t old_pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            size_t new_pmatrix_index = missing_pmatrix_indices.back();

            move.arcRemovalData.remapped_pmatrix_indices.emplace_back(std::make_pair(old_pmatrix_index, new_pmatrix_index));

            // invalidate the pmatrix entry
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                // skip remote partitions
                if (!ann_network.fake_treeinfo->partitions[p]) {
                    continue;
                }
                ann_network.fake_treeinfo->pmatrix_valid[p][old_pmatrix_index] = 0;
            }

            // update all references to this pmatrix index
            ann_network.network.edges[i].pmatrix_index = new_pmatrix_index;
            ann_network.network.edges_by_index[new_pmatrix_index] = &ann_network.network.edges[i];
            ann_network.network.edges_by_index[old_pmatrix_index] = nullptr;

             // also update entries in branch length array
            if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
                for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                    // skip remote partitions
                    if (!ann_network.fake_treeinfo->partitions[p]) {
                        continue;
                    }
                    ann_network.fake_treeinfo->branch_lengths[p][new_pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][old_pmatrix_index];
                    ann_network.fake_treeinfo->branch_lengths[p][old_pmatrix_index] = 0.0;
                }
            } else {
                ann_network.fake_treeinfo->linked_branch_lengths[new_pmatrix_index] = ann_network.fake_treeinfo->linked_branch_lengths[old_pmatrix_index];
                ann_network.fake_treeinfo->linked_branch_lengths[old_pmatrix_index] = 0.0;
            }

            ann_network.network.edges[i].link1->edge_pmatrix_index = new_pmatrix_index;
            ann_network.network.edges[i].link2->edge_pmatrix_index = new_pmatrix_index;

            missing_pmatrix_indices.pop_back();
        }
    }
}

bool assert_links_in_range2(const Network& network) {
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        for (size_t j = 0; j < network.nodes_by_index[i]->links.size(); ++j) {
            assert(network.nodes_by_index[i]->links[j].edge_pmatrix_index < network.num_branches());
        }
    }
    for (size_t i = 0; i < network.num_branches(); ++i) {
        assert(network.edges_by_index[i]->link1->edge_pmatrix_index == i);
        assert(network.edges_by_index[i]->link2->edge_pmatrix_index == i);
    }
    return true;
}

void repairConsecutiveIndices(AnnotatedNetwork &ann_network, Move& move) {
    // TODO: This relabeling procedure will invalidate remaining arc removal move candidates.
    //       This can be circumvented by reloading the network state...

    // ensure that pmatrix indices and clv indices remain consecutive. Do the neccessary relabelings.
    repairConsecutiveClvIndices(ann_network, move);
    repairConsecutivePmatrixIndices(ann_network, move);
}

void revertRemappedClvIndices(AnnotatedNetwork& ann_network, Move& move) {
    if (move.arcRemovalData.remapped_clv_indices.empty()) {
        return;
    }

    for (size_t i = 0; i < move.arcRemovalData.remapped_clv_indices.size(); ++i) {
        size_t old_clv_index = move.arcRemovalData.remapped_clv_indices[i].second;
        size_t new_clv_index = move.arcRemovalData.remapped_clv_indices[i].first;
        Node* node = ann_network.network.nodes_by_index[old_clv_index];

        // invalidate the clv entry
        invalidateSingleClv(ann_network, old_clv_index);

        // update all references to this clv index
        node->clv_index = new_clv_index;
        node->scaler_index = new_clv_index - ann_network.network.num_tips();
        ann_network.network.nodes_by_index[new_clv_index] = node;
        ann_network.network.nodes_by_index[old_clv_index] = nullptr;
        for (size_t j = 0; j < node->links.size(); ++j) {
            node->links[j].node_clv_index = new_clv_index;
        }
    }
}

void revertRemappedPmatrixIndices(AnnotatedNetwork& ann_network, Move& move) {
    if (move.arcRemovalData.remapped_pmatrix_indices.empty()) {
        return;
    }

    for (size_t i = 0; i < move.arcRemovalData.remapped_pmatrix_indices.size(); ++i) {
        size_t old_pmatrix_index = move.arcRemovalData.remapped_pmatrix_indices[i].second;
        size_t new_pmatrix_index = move.arcRemovalData.remapped_pmatrix_indices[i].first;
        Edge* edge = ann_network.network.edges_by_index[old_pmatrix_index];

        // invalidate the pmatrix entry
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            ann_network.fake_treeinfo->pmatrix_valid[p][old_pmatrix_index] = 0;
        }

        // update all references to this pmatrix index
        edge->pmatrix_index = new_pmatrix_index;
        ann_network.network.edges_by_index[new_pmatrix_index] = edge;
        ann_network.network.edges_by_index[old_pmatrix_index] = nullptr;

        // also update entries in branch length array
        if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                // skip remote partitions
                if (!ann_network.fake_treeinfo->partitions[p]) {
                    continue;
                }
                ann_network.fake_treeinfo->branch_lengths[p][new_pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][old_pmatrix_index];
                ann_network.fake_treeinfo->branch_lengths[p][old_pmatrix_index] = 0.0;
            }
        } else {
            ann_network.fake_treeinfo->linked_branch_lengths[new_pmatrix_index] = ann_network.fake_treeinfo->linked_branch_lengths[old_pmatrix_index];
            ann_network.fake_treeinfo->linked_branch_lengths[old_pmatrix_index] = 0.0;
        }

        edge->link1->edge_pmatrix_index = new_pmatrix_index;
        edge->link2->edge_pmatrix_index = new_pmatrix_index;
    }
}

void revertRemappedIndices(AnnotatedNetwork& ann_network, Move& move) {
    revertRemappedClvIndices(ann_network, move);
    revertRemappedPmatrixIndices(ann_network, move);
}

void removeNode(AnnotatedNetwork &ann_network, Node *node) {
    assert(node);
    assert(!node->isTip());
    Network& network = ann_network.network;

    NodeType nodeType = node->type;
    size_t index = node->clv_index;
    size_t index_in_nodes_array = network.nodes_by_index[index] - &network.nodes[0];
    assert(network.nodes[index_in_nodes_array].clv_index == index);

    if (nodeType == NodeType::RETICULATION_NODE) {
        if (network.num_reticulations() > 1) {
            unsigned int node_reticulation_index = node->getReticulationData()->reticulation_index;
            if (node_reticulation_index < network.num_reticulations() - 1) {
                size_t last_reticulation_index = ann_network.network.num_reticulations() - 1;
                // update reticulation indices
                std::swap(ann_network.reticulation_probs[node_reticulation_index], ann_network.reticulation_probs[last_reticulation_index]);
                std::swap(network.reticulation_nodes[node_reticulation_index],
                        network.reticulation_nodes[last_reticulation_index]);
                network.reticulation_nodes[node_reticulation_index]->getReticulationData()->reticulation_index =
                        node_reticulation_index;
            }
        }
        network.reticulation_nodes.resize(network.reticulation_nodes.size() - 1);

        for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
            assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
        }
    }

    network.nodes_by_index[index] = nullptr;
    node->clear();
    network.nodeCount--;
}

std::vector<std::pair<size_t, size_t> > getRemappedReticulationIndices(AnnotatedNetwork& ann_network, const std::vector<size_t>& old_reticulation_clv_indices) {
    std::vector<std::pair<size_t, size_t> > remapped_reticulation_indices;
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        if (ann_network.network.reticulation_nodes[i]->clv_index != old_reticulation_clv_indices[i]) { // a reticulation clv index has been MPI_T_category_changed
            size_t old_ret_index = std::numeric_limits<size_t>::max();
            for (size_t j = 0; j < old_reticulation_clv_indices.size(); ++j) {
                if (old_reticulation_clv_indices[j] == ann_network.network.reticulation_nodes[i]->clv_index) {
                    old_ret_index = j;
                    break;
                }
            }
            if (old_ret_index == std::numeric_limits<size_t>::max()) {
                continue; // this reticulation has been deleted
            }
            size_t new_ret_index = i;
            remapped_reticulation_indices.emplace_back(std::make_pair(old_ret_index, new_ret_index));
        }
    }
    return remapped_reticulation_indices;
}

std::vector<size_t> invertPermutation(const std::vector<size_t>& perm) {
    // https://www.geeksforgeeks.org/inverse-permutation/
    std::vector<size_t> res(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        res[perm[i] - 1] = i + 1;
    }
    return res;
}

std::vector<size_t> permutationToMoveThisReticulationToEnd(AnnotatedNetwork& ann_network, size_t retNodeClvIndex) {
    std::vector<size_t> res(ann_network.network.num_reticulations());
    std::iota(res.begin(), res.end(), 1);
    auto it = std::find_if(ann_network.network.reticulation_nodes.begin(), ann_network.network.reticulation_nodes.end(), 
        [&retNodeClvIndex] (Node* node) { 
            return node->clv_index == retNodeClvIndex; 
        } 
    );
    size_t retPos = std::distance(ann_network.network.reticulation_nodes.begin(), it);
    std::swap(res[res.size() - 1], retPos);
    return res;
}

bool isReticulation(AnnotatedNetwork& ann_network, size_t clvIdx) {
    auto it = std::find_if(ann_network.network.reticulation_nodes.begin(), ann_network.network.reticulation_nodes.end(), 
        [&clvIdx] (Node* node) { 
            return node->clv_index == clvIdx; 
        } 
    );
    return (it != ann_network.network.reticulation_nodes.end());
}

void reorderReticulations(AnnotatedNetwork& ann_network, const std::vector<size_t>& indices, Move& move) {
    assert(indices.size() >= ann_network.network.num_reticulations());
    std::vector<Node*> oldReticulations = ann_network.network.reticulation_nodes;
    std::vector<double> oldReticulationProbs = ann_network.reticulation_probs;

    // https://devblogs.microsoft.com/oldnewthing/20170102-00/?p=95095
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        ann_network.network.reticulation_nodes[i] = oldReticulations[indices[i]-1];
        ann_network.reticulation_probs[i] = oldReticulationProbs[indices[i]-1];
        ann_network.network.reticulation_nodes[i]->getReticulationData()->reticulation_index = i;
    }

    for (NodeDisplayedTreeData& ndtd : ann_network.pernode_displayed_tree_data) {
        for (size_t i = 0; i < ndtd.num_active_displayed_trees; ++i) {
            ReticulationConfigSet& rcs = ndtd.displayed_trees[i].treeLoglData.reticulationChoices;
            for (std::vector<ReticulationState>& config : rcs.configs) {
                std::vector<ReticulationState> tmp_config = config;
                // do the remapping
                for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
                    config[i] = tmp_config[indices[i]-1];
                }
            }
        }
    }

    // update maybe changed clv indices in the move object
    assert(isArcRemoval(move.moveType));
    std::vector<std::pair<size_t, size_t> > remappedIndices = getRemappedReticulationIndices(ann_network, move.arcRemovalData.old_reticulation_clv_indices);

    for (const std::pair<size_t, size_t>& p : remappedIndices) {
        if (move.arcRemovalData.a_clv_index == p.first) {
            move.arcRemovalData.a_clv_index = p.second;
        }
        if (move.arcRemovalData.b_clv_index == p.first) {
            move.arcRemovalData.b_clv_index = p.second;
        }
        if (move.arcRemovalData.c_clv_index == p.first) {
            move.arcRemovalData.c_clv_index = p.second;
        }
        if (move.arcRemovalData.d_clv_index == p.first) {
            move.arcRemovalData.d_clv_index = p.second;
        }
        if (move.arcRemovalData.u_clv_index == p.first) {
            move.arcRemovalData.u_clv_index = p.second;
        }
        if (move.arcRemovalData.v_clv_index == p.first) {
            move.arcRemovalData.v_clv_index = p.second;
        }
    }
}

void reorderReticulationsUndo(AnnotatedNetwork& ann_network, const std::vector<size_t>& indices, Move& move) {
    reorderReticulations(ann_network, invertPermutation(indices), move);
}

void remapReticulationConfigs(AnnotatedNetwork& ann_network, const ArcRemovalData& removalData, bool undo) {
    // TODO: We also need to take remapped clv indices into account here!

    assert(ann_network.reticulation_probs.size() >= ann_network.network.num_reticulations());
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (undo) {
            std::cout << "\nremap reticulation configs undo...\n";
        } else {
            std::cout << "\nremap reticulation configs perform...\n";
        }
    }

    if (undo) {
        assert(removalData.old_reticulation_clv_indices.size() == ann_network.network.num_reticulations());
    } else {
        assert(removalData.old_reticulation_clv_indices.size() == ann_network.network.num_reticulations() + 1);
    }
    std::vector<std::pair<size_t, size_t> > remapped_indices;
    std::vector<Node*> tmp_ret_nodes = ann_network.network.reticulation_nodes;
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        size_t curr_ret_idx = i;
        Node* retNode = tmp_ret_nodes[i];
        for (size_t j = 0; j < removalData.old_reticulation_clv_indices.size(); ++j) {
            if (removalData.old_reticulation_clv_indices[j] == retNode->clv_index) {
                size_t old_ret_idx = j;
                if (curr_ret_idx != old_ret_idx) {
                    remapped_indices.emplace_back(std::make_pair(curr_ret_idx, old_ret_idx));

                    if (undo) {
                        ann_network.network.reticulation_nodes[old_ret_idx] = tmp_ret_nodes[curr_ret_idx];
                        ann_network.network.reticulation_nodes[old_ret_idx]->getReticulationData()->reticulation_index = old_ret_idx;
                        ann_network.reticulation_probs[old_ret_idx] = removalData.old_reticulation_probs[old_ret_idx];
                    } else {
                        ann_network.network.reticulation_nodes[curr_ret_idx] = tmp_ret_nodes[old_ret_idx];
                        ann_network.network.reticulation_nodes[curr_ret_idx]->getReticulationData()->reticulation_index = curr_ret_idx;
                        ann_network.reticulation_probs[curr_ret_idx] = removalData.old_reticulation_probs[old_ret_idx];
                    }
                }
                break;
            }
        }
    }

    for (NodeDisplayedTreeData& ndtd : ann_network.pernode_displayed_tree_data) {
        for (size_t i = 0; i < ndtd.num_active_displayed_trees; ++i) {
            ReticulationConfigSet& rcs = ndtd.displayed_trees[i].treeLoglData.reticulationChoices;
            for (std::vector<ReticulationState>& config : rcs.configs) {
                std::vector<ReticulationState> tmp_config = config;
                // do the remapping
                for (const std::pair<size_t, size_t>& remap_pair : remapped_indices) {
                    size_t curr_ret_idx = remap_pair.first;
                    size_t old_ret_idx = remap_pair.second;
                    if (undo) {
                        config[old_ret_idx] = tmp_config[curr_ret_idx];
                    } else {
                        config[curr_ret_idx] = tmp_config[old_ret_idx];
                    }
                }

            }
        }
    }
}

void performMoveArcRemoval(AnnotatedNetwork &ann_network, Move &move) {
    assert(checkSanityArcRemoval(ann_network, move));
    assert(assert_links_in_range2(ann_network.network));
    assert(assertBranchLengths(ann_network));

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "performing " << toString(move) << "\n";
        std::cout << "reticulation clv indices before perform arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.network.reticulation_nodes[i]->clv_index << "\n";
        }
        std::cout << "reticulation probs before perform arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.reticulation_probs[i] << "\n";
        }
        std::cout << "displayed tree at root configs before perform arc removal:\n";
        for (size_t k = 0; k < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++k) {
            ReticulationConfigSet& rcs = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[k].treeLoglData.reticulationChoices;
            printReticulationChoices(rcs);
        }
    }

    move.arcRemovalData.old_reticulation_probs = ann_network.reticulation_probs;
    move.arcRemovalData.old_reticulation_clv_indices.resize(ann_network.network.num_reticulations());
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        move.arcRemovalData.old_reticulation_clv_indices[i] = ann_network.network.reticulation_nodes[i]->clv_index;
    }
    size_t old_num_nodes = ann_network.network.num_nodes();

    assert(move.moveType == MoveType::ArcRemovalMove || move.moveType == MoveType::DeltaMinusMove);
    assert(assertConsecutiveIndices(ann_network));

    Network &network = ann_network.network;
    Link *from_a_link = getLinkToNode(network, move.arcRemovalData.a_clv_index, move.arcRemovalData.u_clv_index);
    Link *to_b_link = getLinkToNode(network, move.arcRemovalData.b_clv_index, move.arcRemovalData.u_clv_index);
    Link *from_c_link = getLinkToNode(network, move.arcRemovalData.c_clv_index, move.arcRemovalData.v_clv_index);
    Link *to_d_link = getLinkToNode(network, move.arcRemovalData.d_clv_index, move.arcRemovalData.v_clv_index);
    assert(from_a_link);
    assert(to_b_link);
    assert(from_c_link);
    assert(to_d_link);

    std::vector<double> a_b_edge_length = move.arcRemovalData.a_b_len;
    std::vector<double> c_d_edge_length = move.arcRemovalData.c_d_len;

    assert(getEdgeTo(network, move.arcRemovalData.a_clv_index, move.arcRemovalData.u_clv_index));
    assert(getEdgeTo(network, move.arcRemovalData.u_clv_index, move.arcRemovalData.b_clv_index));
    assert(getEdgeTo(network, move.arcRemovalData.c_clv_index, move.arcRemovalData.v_clv_index));
    assert(getEdgeTo(network, move.arcRemovalData.v_clv_index, move.arcRemovalData.d_clv_index));
    assert(getEdgeTo(network, move.arcRemovalData.u_clv_index, move.arcRemovalData.v_clv_index));
    assert(network.num_reticulations() <= network.reticulation_nodes.size());

    size_t a_u_edge_index = getEdgeTo(network, move.arcRemovalData.a_clv_index, move.arcRemovalData.u_clv_index)->pmatrix_index;
    size_t u_b_edge_index = getEdgeTo(network, move.arcRemovalData.u_clv_index, move.arcRemovalData.b_clv_index)->pmatrix_index;
    size_t c_v_edge_index = getEdgeTo(network, move.arcRemovalData.c_clv_index, move.arcRemovalData.v_clv_index)->pmatrix_index;
    size_t v_d_edge_index = getEdgeTo(network, move.arcRemovalData.v_clv_index, move.arcRemovalData.d_clv_index)->pmatrix_index;
    size_t u_v_edge_index = getEdgeTo(network, move.arcRemovalData.u_clv_index, move.arcRemovalData.v_clv_index)->pmatrix_index;
    
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
    }
    removeNode(ann_network, network.nodes_by_index[move.arcRemovalData.u_clv_index]);
    removeNode(ann_network, network.nodes_by_index[move.arcRemovalData.v_clv_index]);
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
    }

    assert(a_u_edge_index < network.edges_by_index.size());
    assert(u_b_edge_index < network.edges_by_index.size());
    assert(c_v_edge_index < network.edges_by_index.size());
    assert(v_d_edge_index < network.edges_by_index.size());
    assert(u_v_edge_index < network.edges_by_index.size());
    removeEdge(ann_network, network.edges_by_index[a_u_edge_index]);
    removeEdge(ann_network, network.edges_by_index[u_b_edge_index]);
    removeEdge(ann_network, network.edges_by_index[c_v_edge_index]);
    removeEdge(ann_network, network.edges_by_index[v_d_edge_index]);
    removeEdge(ann_network, network.edges_by_index[u_v_edge_index]);

    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
    }

    Edge *a_b_edge = addEdge(ann_network, from_a_link, to_b_link, a_b_edge_length[0],
            move.arcRemovalData.wanted_ab_pmatrix_index);
    Edge *c_d_edge = addEdge(ann_network, from_c_link, to_d_link, c_d_edge_length[0],
            move.arcRemovalData.wanted_cd_pmatrix_index);
    assert(c_d_edge);
    assert(a_b_edge);
    set_edge_lengths(ann_network, a_b_edge->pmatrix_index, a_b_edge_length);
    set_edge_lengths(ann_network, c_d_edge->pmatrix_index, c_d_edge_length);
    move.arcRemovalData.wanted_ab_pmatrix_index = a_b_edge->pmatrix_index;
    move.arcRemovalData.wanted_cd_pmatrix_index = c_d_edge->pmatrix_index;

    assert(ann_network.network.nodes_by_index[move.arcRemovalData.u_clv_index] == nullptr);
    assert(ann_network.network.nodes_by_index[move.arcRemovalData.v_clv_index] == nullptr);
    assert(old_num_nodes == ann_network.network.num_nodes() + 2);

    repairConsecutiveIndices(ann_network, move);

    assert(move.arcRemovalData.b_clv_index < network.nodes_by_index.size());
    Node *b = network.nodes_by_index[move.arcRemovalData.b_clv_index];
    assert(b);
    if (b->type == NodeType::RETICULATION_NODE) {
        // u is no longer parent of b, but a is now the parent
        Link *badToParentLink = nullptr;
        assert(b->getReticulationData());
        if (getReticulationFirstParentPmatrixIndex(b) == u_b_edge_index) {
            badToParentLink = b->getReticulationData()->link_to_first_parent;
        } else {
            assert(getReticulationSecondParentPmatrixIndex(b) == u_b_edge_index);
            badToParentLink = b->getReticulationData()->link_to_second_parent;
        }
        assert(badToParentLink);
        badToParentLink->outer = from_a_link;
        badToParentLink->outer->outer = badToParentLink;
    }

    assert(move.arcRemovalData.d_clv_index < network.nodes_by_index.size());
    Node *d = network.nodes_by_index[move.arcRemovalData.d_clv_index];
    assert(d);
    if (d->type == NodeType::RETICULATION_NODE) {
        // v is no longer parent of d, but c is now the parent
        Link *badToParentLink = nullptr;
        assert(d->getReticulationData());
        if (getReticulationFirstParentPmatrixIndex(d) == v_d_edge_index) {
            badToParentLink = d->getReticulationData()->link_to_first_parent;
        } else {
            assert(getReticulationSecondParentPmatrixIndex(d) == v_d_edge_index);
            badToParentLink = d->getReticulationData()->link_to_second_parent;
        }
        assert(badToParentLink);
        badToParentLink->outer = from_c_link;
        badToParentLink->outer->outer = badToParentLink;
    }
    std::vector<size_t> updateMe;
    if (c_d_edge) {
        updateMe = { a_b_edge->pmatrix_index, c_d_edge->pmatrix_index };
    } else {
        updateMe = { a_b_edge->pmatrix_index };
    }
    invalidate_pmatrices(ann_network, updateMe);

    from_a_link->outer = to_b_link;
    to_b_link->outer = from_a_link;
    from_c_link->outer = to_d_link;
    to_d_link->outer = from_c_link;
    from_a_link->edge_pmatrix_index = a_b_edge->pmatrix_index;
    to_b_link->edge_pmatrix_index = a_b_edge->pmatrix_index;
    from_c_link->edge_pmatrix_index = c_d_edge->pmatrix_index;
    to_d_link->edge_pmatrix_index = c_d_edge->pmatrix_index;

    fixReticulationsArcRemoval(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.arcRemovalData.a_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.arcRemovalData.b_clv_index], false, visited);
    invalidatePmatrixIndex(ann_network, a_b_edge->pmatrix_index, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.arcRemovalData.c_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.arcRemovalData.d_clv_index], false, visited);
    invalidatePmatrixIndex(ann_network, c_d_edge->pmatrix_index, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    assert(checkSanity(network));
    assert(assertReticulationProbs(ann_network));
    assert(assertConsecutiveIndices(ann_network));

    bool all_clvs_valid = true;
    for (size_t i = 0; i < ann_network.fake_treeinfo->partition_count; ++i) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[i]) {
            continue;
        }
        all_clvs_valid &= ann_network.fake_treeinfo->clv_valid[i][ann_network.network.root->clv_index];
    }
    assert(!all_clvs_valid);

    assert(assert_links_in_range2(ann_network.network));
    assert(assertBranchLengths(ann_network));

    remapReticulationConfigs(ann_network, move.arcRemovalData, false);

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "reticulation clv indices after perform arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.network.reticulation_nodes[i]->clv_index << "\n";
        }
        std::cout << "reticulation probs after perform arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.reticulation_probs[i] << "\n";
        }
        std::cout << "displayed tree at root configs after perform arc removal:\n";
        for (size_t k = 0; k < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++k) {
            ReticulationConfigSet& rcs = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[k].treeLoglData.reticulationChoices;
            printReticulationChoices(rcs);
        }
    }
}

void undoMoveArcRemoval(AnnotatedNetwork &ann_network, Move &move) {
    assert(move.moveType == MoveType::ArcRemovalMove || move.moveType == MoveType::DeltaMinusMove);
    assert(assertConsecutiveIndices(ann_network));
    assert(assertBranchLengths(ann_network));

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "undoing " << toString(move) << "\n";
        std::cout << "reticulation clv indices before undo arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.network.reticulation_nodes[i]->clv_index << "\n";
        }
        std::cout << "reticulation probs before undo arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.reticulation_probs[i] << "\n";
        }
        std::cout << "displayed tree at root configs before undo arc removal:\n";
        for (size_t k = 0; k < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++k) {
            ReticulationConfigSet& rcs = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[k].treeLoglData.reticulationChoices;
            printReticulationChoices(rcs);
        }
    }

    /*if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "undo " << toString(move) << "\n";
        std::cout << "remapped clv indices:\n";
        for (size_t i = 0; i < move.remapped_clv_indices.size(); ++i) {
            std::cout << " " << move.remapped_clv_indices[i].first << " -> " << move.remapped_clv_indices[i].second << "\n";
        }
        std::cout << "remapped pmatrix indices:\n";
        for (size_t i = 0; i < move.remapped_pmatrix_indices.size(); ++i) {
            std::cout << " " << move.remapped_pmatrix_indices[i].first << " -> " << move.remapped_pmatrix_indices[i].second << "\n";
        }
        //std::cout << exportDebugInfo(ann_network) << "\n";
    }*/

    revertRemappedIndices(ann_network, move);

    Move insertion = buildMoveArcInsertion(move.arcRemovalData.a_clv_index, move.arcRemovalData.b_clv_index,
            move.arcRemovalData.c_clv_index, move.arcRemovalData.d_clv_index, move.arcRemovalData.u_v_len, move.arcRemovalData.c_v_len, move.arcRemovalData.a_u_len, move.arcRemovalData.a_b_len, move.arcRemovalData.c_d_len, move.arcRemovalData.v_d_len, move.arcRemovalData.u_b_len, MoveType::ArcInsertionMove, move.edge_orig_idx, move.node_orig_idx);

    insertion.arcInsertionData.wanted_u_clv_index = move.arcRemovalData.u_clv_index;
    insertion.arcInsertionData.wanted_v_clv_index = move.arcRemovalData.v_clv_index;

    insertion.arcInsertionData.wanted_au_pmatrix_index = move.arcRemovalData.au_pmatrix_index;
    insertion.arcInsertionData.wanted_ub_pmatrix_index = move.arcRemovalData.ub_pmatrix_index;
    insertion.arcInsertionData.wanted_cv_pmatrix_index = move.arcRemovalData.cv_pmatrix_index;
    insertion.arcInsertionData.wanted_vd_pmatrix_index = move.arcRemovalData.vd_pmatrix_index;
    insertion.arcInsertionData.wanted_uv_pmatrix_index = move.arcRemovalData.uv_pmatrix_index;

    insertion.arcInsertionData.ab_pmatrix_index = move.arcRemovalData.wanted_ab_pmatrix_index;
    insertion.arcInsertionData.cd_pmatrix_index = move.arcRemovalData.wanted_cd_pmatrix_index;

    performMoveArcInsertion(ann_network, insertion);
    assert(assertConsecutiveIndices(ann_network));
    assert(assertBranchLengths(ann_network));

    std::vector<std::pair<size_t, size_t> > remapped_reticulation_indices = getRemappedReticulationIndices(ann_network, move.arcRemovalData.old_reticulation_clv_indices);
    remapReticulationConfigs(ann_network, move.arcRemovalData, true);
    ann_network.reticulation_probs = move.arcRemovalData.old_reticulation_probs;

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "reticulation clv indices after undo arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.network.reticulation_nodes[i]->clv_index << "\n";
        }
        std::cout << "reticulation probs after undo arc removal:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << ann_network.reticulation_probs[i] << "\n";
        }
        std::cout << "displayed tree at root configs after undo arc removal:\n";
        for (size_t k = 0; k < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++k) {
            ReticulationConfigSet& rcs = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[k].treeLoglData.reticulationChoices;
            printReticulationChoices(rcs);
        }
    }
}

std::string toStringArcRemoval(const Move &move) {
    std::stringstream ss;
    ss << "arc removal move:\n";
    ss << "  a = " << move.arcRemovalData.a_clv_index << "\n";
    ss << "  b = " << move.arcRemovalData.b_clv_index << "\n";
    ss << "  c = " << move.arcRemovalData.c_clv_index << "\n";
    ss << "  d = " << move.arcRemovalData.d_clv_index << "\n";
    ss << "  u = " << move.arcRemovalData.u_clv_index << "\n";
    ss << "  v = " << move.arcRemovalData.v_clv_index << "\n";
    ss << "  au = " << move.arcRemovalData.au_pmatrix_index << "\n";
    ss << "   a_u_len: " << move.arcRemovalData.a_u_len << "\n";
    ss << "  cv = " << move.arcRemovalData.cv_pmatrix_index << "\n";
    ss << "   c_v_len: " << move.arcRemovalData.c_v_len << "\n";
    ss << "  ub = " << move.arcRemovalData.ub_pmatrix_index << "\n";
    ss << "   u_b_len: " << move.arcRemovalData.u_b_len << "\n";
    ss << "  vd = " << move.arcRemovalData.vd_pmatrix_index << "\n";
    ss << "   v_d_len: " << move.arcRemovalData.v_d_len << "\n";
    ss << "  uv = " << move.arcRemovalData.uv_pmatrix_index << "\n";
    ss << "   u_v_len: " << move.arcRemovalData.u_v_len << "\n";
    ss << "  wanted ab = " << move.arcRemovalData.wanted_ab_pmatrix_index << "\n";
    ss << "   a_b_len: " << move.arcRemovalData.a_b_len << "\n";
    ss << "  wanted cd = " << move.arcRemovalData.wanted_cd_pmatrix_index << "\n";
    ss << "   c_d_len: " << move.arcRemovalData.c_d_len << "\n"; 
    return ss.str();
}

std::unordered_set<size_t> brlenOptCandidatesArcRemoval(AnnotatedNetwork &ann_network, const Move &move) {
    Move remapped_move(move);
    for (size_t i = 0; i < move.arcRemovalData.remapped_clv_indices.size(); ++i) {
        size_t old_clv_index = move.arcRemovalData.remapped_clv_indices[i].first;
        size_t new_clv_index = move.arcRemovalData.remapped_clv_indices[i].second;
        updateMoveClvIndex(remapped_move, old_clv_index, new_clv_index);
    }
    Node *a = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.a_clv_index];
    Node *b = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.b_clv_index];
    Node *c = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.c_clv_index];
    Node *d = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.d_clv_index];
    Edge *a_b_edge = getEdgeTo(ann_network.network, a, b);
    Edge *c_d_edge = getEdgeTo(ann_network.network, c, d);
    return {a_b_edge->pmatrix_index, c_d_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidatesUndoArcRemoval(AnnotatedNetwork &ann_network,
        const Move &move) {
    Move remapped_move(move);
    for (size_t i = 0; i < move.arcRemovalData.remapped_clv_indices.size(); ++i) {
        size_t old_clv_index = move.arcRemovalData.remapped_clv_indices[i].first;
        size_t new_clv_index = move.arcRemovalData.remapped_clv_indices[i].second;
        updateMoveClvIndex(remapped_move, old_clv_index, new_clv_index);
    }
    Node *a = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.a_clv_index];
    Node *b = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.b_clv_index];
    Node *c = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.c_clv_index];
    Node *d = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.d_clv_index];
    Node *u = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.u_clv_index];
    Node *v = ann_network.network.nodes_by_index[remapped_move.arcRemovalData.v_clv_index];
    Edge *a_u_edge = getEdgeTo(ann_network.network, a, u);
    Edge *u_b_edge = getEdgeTo(ann_network.network, u, b);
    Edge *c_v_edge = getEdgeTo(ann_network.network, c, v);
    Edge *v_d_edge = getEdgeTo(ann_network.network, v, d);
    Edge *u_v_edge = getEdgeTo(ann_network.network, u, v);
    return {a_u_edge->pmatrix_index, u_b_edge->pmatrix_index,c_v_edge->pmatrix_index,v_d_edge->pmatrix_index,u_v_edge->pmatrix_index};
}

Move randomMoveArcRemoval(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    auto moves = possibleMovesArcRemoval(ann_network);
    if (!moves.empty()) {
        return moves[getRandomIndex(ann_network.rng, moves.size())];
    } else {
        throw std::runtime_error("No possible move found");
    }
}

Move randomMoveDeltaMinus(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    auto moves = possibleMovesDeltaMinus(ann_network);
    if (!moves.empty()) {
        return moves[getRandomIndex(ann_network.rng, moves.size())];
    } else {
        throw std::runtime_error("No possible move found");
    }
}

}