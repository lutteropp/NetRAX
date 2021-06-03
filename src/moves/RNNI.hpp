#pragma once

#include "Move.hpp"
struct Edge;

#include <limits>
#include <vector>

namespace netrax {

bool checkSanityRNNI(AnnotatedNetwork &ann_network, const Move &move);
std::vector<Move> possibleMovesRNNI(AnnotatedNetwork &ann_network,
                                    const Edge *edge);
std::vector<Move> possibleMovesRNNI(AnnotatedNetwork &ann_network);

std::vector<Move> possibleMovesRNNI(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesRNNI(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

void performMoveRNNI(AnnotatedNetwork &ann_network, Move &move);
void undoMoveRNNI(AnnotatedNetwork &ann_network, Move &move);
std::string toStringRNNI(const Move &move);
std::unordered_set<size_t> brlenOptCandidatesRNNI(AnnotatedNetwork &ann_network,
                                                  const Move &move);
std::unordered_set<size_t> brlenOptCandidatesUndoRNNI(
    AnnotatedNetwork &ann_network, const Move &move);
Move randomMoveRNNI(AnnotatedNetwork &ann_network);
std::vector<RNNIMoveType> validMoveTypes(AnnotatedNetwork &ann_network, Node *u,
                                         Node *v, Node *s, Node *t);

void filterOutDuplicateMovesRNNI(std::vector<Move> &moves);
void updateMoveClvIndexRNNI(Move &move, size_t old_clv_index,
                            size_t new_clv_index, bool undo);

}  // namespace netrax
