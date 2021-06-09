#include "ParentChange.hpp"

namespace netrax {

bool checkSanityParentChange(AnnotatedNetwork &ann_network, const Move &move);

std::vector<Move> possibleMovesParentChange(AnnotatedNetwork &ann_network,
                                            const Edge *edge, bool noDeltaPlus,
                                            int min_radius, int max_radius);
std::vector<Move> possibleMovesParentChange(AnnotatedNetwork &ann_network,
                                            bool noDeltaPlus, int min_radius,
                                            int max_radius);

std::vector<Move> possibleMovesParentChange(AnnotatedNetwork &ann_network,
                                            const Node *node, bool noDeltaPlus,
                                            int min_radius, int max_radius);

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    bool noDeltaPlus, int min_radius, int max_radius);

std::vector<Move> possibleMovesParentChange(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    bool noDeltaPlus, int min_radius, int max_radius);

void performMoveParentChange(AnnotatedNetwork &ann_network, Move &move) {
  assert(isParentChange(move.moveType));

  Move removal(move);
  removal.moveType = MoveType::ArcRemovalMove;
  performMove(ann_network, removal);

  Move insertion(move);
  insertion.moveType = MoveType::ArcInsertionMove;
  updateMove(ann_network.network, removal, insertion);
  performMove(ann_network, insertion);

  move.arcRemovalData = removal.arcRemovalData;
  move.arcInsertionData = insertion.arcInsertionData;

  move.remapped_clv_indices = insertion.remapped_clv_indices;
  move.remapped_pmatrix_indices = insertion.remapped_pmatrix_indices;
  move.remapped_reticulation_indices = insertion.remapped_reticulation_indices;
}

void undoMoveParentChange(AnnotatedNetwork &ann_network, Move &move) {
  assert(isParentChange(move.moveType));

  Move insertion(move);
  insertion.moveType = MoveType::ArcInsertionMove;
  performMove(ann_network, insertion);

  Move removal(move);
  removal.moveType = MoveType::ArcRemovalMove;
  updateMove(ann_network.network, insertion, removal);
  performMove(ann_network, removal);

  move.arcRemovalData = removal.arcRemovalData;
  move.arcInsertionData = insertion.arcInsertionData;

  move.remapped_clv_indices = removal.remapped_clv_indices;
  move.remapped_pmatrix_indices = removal.remapped_pmatrix_indices;
  move.remapped_reticulation_indices = removal.remapped_reticulation_indices;
}

std::string toStringParentChange(const Move &move);

std::unordered_set<size_t> brlenOptCandidatesParentChange(
    AnnotatedNetwork &ann_network, const Move &move);

std::unordered_set<size_t> brlenOptCandidatesUndoParentChange(
    AnnotatedNetwork &ann_network, const Move &move);

Move randomMoveParentChange(AnnotatedNetwork &ann_network);

void updateMoveClvIndexParentChange(Move &move, size_t old_clv_index,
                                    size_t new_clv_index, bool undo);
void updateMovePmatrixIndexParentChange(Move &move, size_t old_pmatrix_index,
                                        size_t new_pmatrix_index, bool undo);

}  // namespace netrax