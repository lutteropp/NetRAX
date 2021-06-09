#include "ParentChange.hpp"

namespace netrax {

bool checkSanityParentChange(AnnotatedNetwork &ann_network, const Move &move) {
  throw std::runtime_error("Not implemented yet");
}

std::vector<Move> possibleMovesParentChange(AnnotatedNetwork &ann_network,
                                            const Edge *edge, bool noDeltaPlus,
                                            int min_radius, int max_radius) {
  throw std::runtime_error("Not implemented yet");
}

std::vector<Move> possibleMovesParentChange(AnnotatedNetwork &ann_network,
                                            bool noDeltaPlus, int min_radius,
                                            int max_radius) {
  throw std::runtime_error("Not implemented yet");
}

std::vector<Move> possibleMovesParentChange(AnnotatedNetwork &ann_network,
                                            const Node *node, bool noDeltaPlus,
                                            int min_radius, int max_radius) {
  throw std::runtime_error("Not implemented yet");
}

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    bool noDeltaPlus, int min_radius, int max_radius) {
  throw std::runtime_error("Not implemented yet");
}

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    bool noDeltaPlus, int min_radius, int max_radius) {
  throw std::runtime_error("Not implemented yet");
}

void performMoveParentChange(AnnotatedNetwork &ann_network, Move &move) {
  throw std::runtime_error("Not implemented yet");
}

void undoMoveParentChange(AnnotatedNetwork &ann_network, Move &move) {
  throw std::runtime_error("Not implemented yet");
}

std::string toStringParentChange(const Move &move) {
  throw std::runtime_error("Not implemented yet");
}

std::unordered_set<size_t> brlenOptCandidatesParentChange(
    AnnotatedNetwork &ann_network, const Move &move) {
  throw std::runtime_error("Not implemented yet");
}

std::unordered_set<size_t> brlenOptCandidatesUndoParentChange(
    AnnotatedNetwork &ann_network, const Move &move) {
  throw std::runtime_error("Not implemented yet");
}

Move randomMoveParentChange(AnnotatedNetwork &ann_network) {
  throw std::runtime_error("Not implemented yet");
}

void updateMoveClvIndexParentChange(Move &move, size_t old_clv_index,
                                    size_t new_clv_index, bool undo) {
  throw std::runtime_error("Not implemented yet");
}

void updateMovePmatrixIndexParentChange(Move &move, size_t old_pmatrix_index,
                                        size_t new_pmatrix_index, bool undo) {
  throw std::runtime_error("Not implemented yet");
}

}  // namespace netrax