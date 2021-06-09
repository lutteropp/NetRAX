#pragma once

#include "Move.hpp"
struct Edge;

namespace netrax {

bool checkSanityParentChange(AnnotatedNetwork &ann_network, const Move &move);

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const Edge *edge, bool noDeltaPlus = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, bool noDeltaPlus = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const Node *node, bool noDeltaPlus = false,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    bool noDeltaPlus = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    bool noDeltaPlus = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());

void performMoveParentChange(AnnotatedNetwork &ann_network, Move &move);

void undoMoveParentChange(AnnotatedNetwork &ann_network, Move &move);
std::string toStringParentChange(const Move &move);

std::unordered_set<size_t> brlenOptCandidatesParentChange(
    AnnotatedNetwork &ann_network, const Move &move);

std::unordered_set<size_t> brlenOptCandidatesUndoParentChange(
    AnnotatedNetwork &ann_network, const Move &move);

Move randomMoveParentChange(AnnotatedNetwork &ann_network);

void updateMoveClvIndexParentChange(Move &move, size_t old_clv_index,
                                    size_t new_clv_index, bool undo = false);
void updateMovePmatrixIndexParentChange(Move &move, size_t old_pmatrix_index,
                                        size_t new_pmatrix_index,
                                        bool undo = false);

}  // namespace netrax