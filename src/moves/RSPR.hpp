#pragma once

#include "Move.hpp"
struct Edge;

#include <limits>
#include <vector>

namespace netrax {

bool checkSanityRSPR(AnnotatedNetwork &ann_network, const Move &move);

std::vector<Move> possibleMovesRSPR(
    AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesRSPR1(
    AnnotatedNetwork &ann_network, const Edge *edge, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesRSPR(
    AnnotatedNetwork &ann_network, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesRSPR1(
    AnnotatedNetwork &ann_network, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesTail(
    AnnotatedNetwork &ann_network, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesHead(
    AnnotatedNetwork &ann_network, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesTail(
    AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesHead(
    AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesRSPR(
    AnnotatedNetwork &ann_network, const Node *node, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesRSPR1(
    AnnotatedNetwork &ann_network, const Node *node, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesTail(
    AnnotatedNetwork &ann_network, const Node *node, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesHead(
    AnnotatedNetwork &ann_network, const Node *node, bool noRSPR1Moves = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesRSPR(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    bool noRSPR1Moves = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesRSPR1(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesTail(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    bool noRSPR1Moves = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesHead(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    bool noRSPR1Moves = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesRSPR(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    bool noRSPR1Moves = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesRSPR1(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesTail(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    bool noRSPR1Moves = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesHead(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    bool noRSPR1Moves = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());

void performMoveRSPR(AnnotatedNetwork &ann_network, Move &move);
void undoMoveRSPR(AnnotatedNetwork &ann_network, Move &move);

std::string toStringRSPR(const Move &move);
std::unordered_set<size_t> brlenOptCandidatesRSPR(AnnotatedNetwork &ann_network,
                                                  const Move &move);

std::unordered_set<size_t> brlenOptCandidatesUndoRSPR(
    AnnotatedNetwork &ann_network, const Move &move);
Move randomMoveRSPR(AnnotatedNetwork &ann_network);
Move randomMoveRSPR1(AnnotatedNetwork &ann_network);
Move randomMoveTail(AnnotatedNetwork &ann_network);
Move randomMoveHead(AnnotatedNetwork &ann_network);

void updateMoveClvIndexRSPR(Move &move, size_t old_clv_index,
                            size_t new_clv_index, bool undo);

}  // namespace netrax