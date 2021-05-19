#pragma once

#include "Move.hpp"
struct Edge;

#include <vector>
#include <limits>

namespace netrax {

bool checkSanityRNNI(AnnotatedNetwork& ann_network, const Move& move);
std::vector<Move> possibleMovesRNNI(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<Move> possibleMovesRNNI(AnnotatedNetwork &ann_network);

std::vector<Move> possibleMovesRNNI(AnnotatedNetwork& ann_network, const std::vector<Node*>& start_nodes, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMovesRNNI(AnnotatedNetwork& ann_network, const std::vector<Edge*>& start_edges, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

void performMoveRNNI(AnnotatedNetwork &ann_network, Move &move);
void undoMoveRNNI(AnnotatedNetwork &ann_network, Move &move);
std::string toStringRNNI(const Move &move);
std::unordered_set<size_t> brlenOptCandidatesRNNI(AnnotatedNetwork &ann_network, const Move &move);
std::unordered_set<size_t> brlenOptCandidatesUndoRNNI(AnnotatedNetwork &ann_network, const Move &move);
Move randomMoveRNNI(AnnotatedNetwork &ann_network);

}
