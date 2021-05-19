#pragma once

#include "Move.hpp"
struct Edge;

namespace netrax {

bool checkSanityArcInsertion(AnnotatedNetwork& ann_network, const Move& move);

std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
        const Edge *edge, bool noDeltaPlus = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network, bool noDeltaPlus = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
        const Edge *edge, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());


std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
        const Node *node, bool noDeltaPlus = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
        const Node *node, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
        const std::vector<Node*>& start_nodes, bool noDeltaPlus = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
        const std::vector<Node*>& start_nodes, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
        const std::vector<Edge*>& start_edges, bool noDeltaPlus = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
        const std::vector<Edge*>& start_edges, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

void performMoveArcInsertion(AnnotatedNetwork &ann_network, Move &move);

void undoMoveArcInsertion(AnnotatedNetwork &ann_network, Move &move);
std::string toStringArcInsertion(const Move &move);

std::unordered_set<size_t> brlenOptCandidatesArcInsertion(AnnotatedNetwork &ann_network,
        const Move &move);

std::unordered_set<size_t> brlenOptCandidatesUndoArcInsertion(AnnotatedNetwork &ann_network,
        const Move &move);

Move randomMoveArcInsertion(AnnotatedNetwork &ann_network);
Move randomMoveDeltaPlus(AnnotatedNetwork &ann_network);

Move buildMoveArcInsertion(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
        std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType, size_t edge_orig_idx, size_t node_orig_idx);

}