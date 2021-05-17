#pragma once

#include "Move.hpp"
struct Node;

#include <vector>
#include <limits>

namespace netrax {

bool checkSanityArcRemoval(AnnotatedNetwork& ann_network, const Move& move);

std::vector<Move> possibleMovesArcRemoval(AnnotatedNetwork &ann_network, Node *v, size_t edge_orig_idx,
        MoveType moveType = MoveType::ArcRemovalMove);
std::vector<Move> possibleMovesArcRemoval(AnnotatedNetwork &ann_network);

std::vector<Move> possibleMovesDeltaMinus(AnnotatedNetwork &ann_network, Node *v, size_t edge_orig_idx);
std::vector<Move> possibleMovesDeltaMinus(AnnotatedNetwork &ann_network);

std::vector<Move> possibleMovesArcRemoval(AnnotatedNetwork& ann_network, const std::vector<Node*>& start_nodes);

void performMoveArcRemoval(AnnotatedNetwork &ann_network, Move &move);
void undoMoveArcRemoval(AnnotatedNetwork &ann_network, Move &move);

std::string toStringArcRemoval(const Move &move);

std::unordered_set<size_t> brlenOptCandidatesArcRemoval(AnnotatedNetwork &ann_network, const Move &move);

std::unordered_set<size_t> brlenOptCandidatesUndoArcRemoval(AnnotatedNetwork &ann_network,
        const Move &move);

Move randomMoveArcRemoval(AnnotatedNetwork &ann_network);
Move randomMoveDeltaMinus(AnnotatedNetwork &ann_network);

Move buildMoveArcRemoval(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, size_t u_clv_index, size_t v_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
         std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType, size_t edge_orig_idx, size_t node_orig_idx);

}