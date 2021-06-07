#include "Move.hpp"

#include <algorithm>

#include "ArcInsertion.hpp"
#include "ArcRemoval.hpp"
#include "RNNI.hpp"
#include "RSPR.hpp"

#include "GeneralMoveFunctions.hpp"

#include "../helper/Helper.hpp"

namespace netrax {

Move randomMove(AnnotatedNetwork &ann_network, MoveType type) {
  switch (type) {
    case MoveType::ArcInsertionMove:
      return randomMoveArcInsertion(ann_network);
    case MoveType::DeltaPlusMove:
      return randomMoveDeltaPlus(ann_network);
    case MoveType::ArcRemovalMove:
      return randomMoveArcRemoval(ann_network);
    case MoveType::DeltaMinusMove:
      return randomMoveDeltaMinus(ann_network);
    case MoveType::RNNIMove:
      return randomMoveRNNI(ann_network);
    case MoveType::RSPRMove:
      return randomMoveRSPR(ann_network);
    case MoveType::RSPR1Move:
      return randomMoveRSPR1(ann_network);
    case MoveType::HeadMove:
      return randomMoveHead(ann_network);
    case MoveType::TailMove:
      return randomMoveTail(ann_network);
    default:
      throw std::runtime_error("Invalid move type randomMove: " +
                               toString(type));
  }
}

void performMove(AnnotatedNetwork &ann_network, Move &move) {
  switch (move.moveType) {
    case MoveType::ArcInsertionMove:
      performMoveArcInsertion(ann_network, move);
      break;
    case MoveType::DeltaPlusMove:
      performMoveArcInsertion(ann_network, move);
      break;
    case MoveType::ArcRemovalMove:
      performMoveArcRemoval(ann_network, move);
      break;
    case MoveType::DeltaMinusMove:
      performMoveArcRemoval(ann_network, move);
      break;
    case MoveType::RNNIMove:
      performMoveRNNI(ann_network, move);
      break;
    case MoveType::RSPRMove:
      performMoveRSPR(ann_network, move);
      break;
    case MoveType::RSPR1Move:
      performMoveRSPR(ann_network, move);
      break;
    case MoveType::HeadMove:
      performMoveRSPR(ann_network, move);
      break;
    case MoveType::TailMove:
      performMoveRSPR(ann_network, move);
      break;
    default:
      throw std::runtime_error("Invalid move type performMove: " +
                               toString(move.moveType));
      break;
  }
}

void undoMove(AnnotatedNetwork &ann_network, Move &move) {
  switch (move.moveType) {
    case MoveType::ArcInsertionMove:
      undoMoveArcInsertion(ann_network, move);
      break;
    case MoveType::DeltaPlusMove:
      undoMoveArcInsertion(ann_network, move);
      break;
    case MoveType::ArcRemovalMove:
      undoMoveArcRemoval(ann_network, move);
      break;
    case MoveType::DeltaMinusMove:
      undoMoveArcRemoval(ann_network, move);
      break;
    case MoveType::RNNIMove:
      undoMoveRNNI(ann_network, move);
      break;
    case MoveType::RSPRMove:
      undoMoveRSPR(ann_network, move);
      break;
    case MoveType::RSPR1Move:
      undoMoveRSPR(ann_network, move);
      break;
    case MoveType::HeadMove:
      undoMoveRSPR(ann_network, move);
      break;
    case MoveType::TailMove:
      undoMoveRSPR(ann_network, move);
      break;
    default:
      throw std::runtime_error("Invalid move type undoMove: " +
                               toString(move.moveType));
      break;
  }
}

std::string toString(const Move &move) {
  switch (move.moveType) {
    case MoveType::ArcInsertionMove:
      return toStringArcInsertion(move);
    case MoveType::DeltaPlusMove:
      return toStringArcInsertion(move);
    case MoveType::ArcRemovalMove:
      return toStringArcRemoval(move);
    case MoveType::DeltaMinusMove:
      return toStringArcRemoval(move);
    case MoveType::RNNIMove:
      return toStringRNNI(move);
    case MoveType::RSPRMove:
      return toStringRSPR(move);
    case MoveType::RSPR1Move:
      return toStringRSPR(move);
    case MoveType::HeadMove:
      return toStringRSPR(move);
    case MoveType::TailMove:
      return toStringRSPR(move);
    default:
      throw std::runtime_error("Invalid move type toString: " +
                               toString(move.moveType));
  }
}

bool checkSanity(AnnotatedNetwork &ann_network, const Move &move) {
  switch (move.moveType) {
    case MoveType::ArcInsertionMove:
      return checkSanityArcInsertion(ann_network, move);
    case MoveType::DeltaPlusMove:
      return checkSanityArcInsertion(ann_network, move);
    case MoveType::ArcRemovalMove:
      return checkSanityArcRemoval(ann_network, move);
    case MoveType::DeltaMinusMove:
      return checkSanityArcRemoval(ann_network, move);
    case MoveType::RNNIMove:
      return checkSanityRNNI(ann_network, move);
    case MoveType::RSPRMove:
      return checkSanityRSPR(ann_network, move);
    case MoveType::RSPR1Move:
      return checkSanityRSPR(ann_network, move);
    case MoveType::HeadMove:
      return checkSanityRSPR(ann_network, move);
    case MoveType::TailMove:
      return checkSanityRSPR(ann_network, move);
    default:
      throw std::runtime_error("Invalid move type checkSanity: " +
                               toString(move.moveType));
  }
}

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network,
                                              const Move &move) {
  switch (move.moveType) {
    case MoveType::ArcInsertionMove:
      return brlenOptCandidatesArcInsertion(ann_network, move);
    case MoveType::DeltaPlusMove:
      return brlenOptCandidatesArcInsertion(ann_network, move);
    case MoveType::ArcRemovalMove:
      return brlenOptCandidatesArcRemoval(ann_network, move);
    case MoveType::DeltaMinusMove:
      return brlenOptCandidatesArcRemoval(ann_network, move);
    case MoveType::RNNIMove:
      return brlenOptCandidatesRNNI(ann_network, move);
    case MoveType::RSPRMove:
      return brlenOptCandidatesRSPR(ann_network, move);
    case MoveType::RSPR1Move:
      return brlenOptCandidatesRSPR(ann_network, move);
    case MoveType::HeadMove:
      return brlenOptCandidatesRSPR(ann_network, move);
    case MoveType::TailMove:
      return brlenOptCandidatesRSPR(ann_network, move);
    default:
      throw std::runtime_error("Invalid move type brlenOptCandidates: " +
                               toString(move.moveType));
  }
}

std::vector<Move> possibleMoves(AnnotatedNetwork &ann_network, MoveType type,
                                bool rspr1_present, bool delta_plus_present,
                                int min_radius, int max_radius) {
  switch (type) {
    case MoveType::ArcInsertionMove:
      return possibleMovesArcInsertion(ann_network, delta_plus_present,
                                       min_radius, max_radius);
    case MoveType::DeltaPlusMove:
      return possibleMovesDeltaPlus(ann_network, min_radius, max_radius);
    case MoveType::ArcRemovalMove:
      return possibleMovesArcRemoval(ann_network);
    case MoveType::DeltaMinusMove:
      return possibleMovesDeltaMinus(ann_network);
    case MoveType::RNNIMove:
      return possibleMovesRNNI(ann_network);
    case MoveType::RSPR1Move:
      return possibleMovesRSPR1(ann_network, min_radius, max_radius);
    case MoveType::RSPRMove:
      return possibleMovesRSPR(ann_network, rspr1_present, min_radius,
                               max_radius);
    case MoveType::HeadMove:
      return possibleMovesHead(ann_network, rspr1_present, min_radius,
                               max_radius);
    case MoveType::TailMove:
      return possibleMovesTail(ann_network, rspr1_present, min_radius,
                               max_radius);
    default:
      throw std::runtime_error("Invalid move type possibleMoves: " +
                               toString(type));
  }
}

std::vector<Move> possibleMoves(AnnotatedNetwork &ann_network, MoveType type,
                                std::vector<Edge *> start_edges,
                                bool rspr1_present, bool delta_plus_present,
                                int min_radius, int max_radius) {
  for (size_t i = 0; i < start_edges.size(); ++i) {
    assert(start_edges[i]);
  }

  switch (type) {
    case MoveType::ArcInsertionMove:
      return possibleMovesArcInsertion(
          ann_network, start_edges, delta_plus_present, min_radius, max_radius);
    case MoveType::DeltaPlusMove:
      return possibleMovesDeltaPlus(ann_network, start_edges, min_radius,
                                    max_radius);
    case MoveType::ArcRemovalMove:
      return possibleMovesArcRemoval(ann_network, start_edges);
    case MoveType::DeltaMinusMove:
      return possibleMovesDeltaMinus(ann_network, start_edges);
    case MoveType::RNNIMove:
      return possibleMovesRNNI(ann_network, start_edges);
    case MoveType::RSPR1Move:
      return possibleMovesRSPR1(ann_network, start_edges, min_radius,
                                max_radius);
    case MoveType::RSPRMove:
      return possibleMovesRSPR(ann_network, start_edges, rspr1_present,
                               min_radius, max_radius);
    case MoveType::HeadMove:
      return possibleMovesHead(ann_network, start_edges, rspr1_present,
                               min_radius, max_radius);
    case MoveType::TailMove:
      return possibleMovesTail(ann_network, start_edges, rspr1_present,
                               min_radius, max_radius);
    default:
      throw std::runtime_error("Invalid move type possibleMoves: " +
                               toString(type));
  }
}

std::vector<Move> possibleMoves(AnnotatedNetwork &ann_network, MoveType type,
                                std::vector<Node *> start_nodes,
                                bool rspr1_present, bool delta_plus_present,
                                int min_radius, int max_radius) {
  for (size_t i = 0; i < start_nodes.size(); ++i) {
    assert(start_nodes[i]);
  }

  switch (type) {
    case MoveType::ArcInsertionMove:
      return possibleMovesArcInsertion(
          ann_network, start_nodes, delta_plus_present, min_radius, max_radius);
    case MoveType::DeltaPlusMove:
      return possibleMovesDeltaPlus(ann_network, start_nodes, min_radius,
                                    max_radius);
    case MoveType::ArcRemovalMove:
      return possibleMovesArcRemoval(ann_network, start_nodes);
    case MoveType::DeltaMinusMove:
      return possibleMovesDeltaMinus(ann_network, start_nodes);
    case MoveType::RNNIMove:
      return possibleMovesRNNI(ann_network, start_nodes);
    case MoveType::RSPR1Move:
      return possibleMovesRSPR1(ann_network, start_nodes, min_radius,
                                max_radius);
    case MoveType::RSPRMove:
      return possibleMovesRSPR(ann_network, start_nodes, rspr1_present,
                               min_radius, max_radius);
    case MoveType::HeadMove:
      return possibleMovesHead(ann_network, start_nodes, rspr1_present,
                               min_radius, max_radius);
    case MoveType::TailMove:
      return possibleMovesTail(ann_network, start_nodes, rspr1_present,
                               min_radius, max_radius);
    default:
      throw std::runtime_error("Invalid move type possibleMoves: " +
                               toString(type));
  }
}

std::vector<Move> possibleMoves(AnnotatedNetwork &ann_network,
                                std::vector<MoveType> types,
                                std::vector<Edge *> start_edges) {
  std::vector<Move> res;
  std::vector<Move> moreMoves;
  for (MoveType type : types) {
    moreMoves = possibleMoves(ann_network, type, start_edges);
    res.insert(std::end(res), std::begin(moreMoves), std::end(moreMoves));
  }
  return res;
}

std::vector<Move> possibleMoves(AnnotatedNetwork &ann_network,
                                std::vector<MoveType> types,
                                std::vector<Node *> start_nodes) {
  std::vector<Move> res;
  std::vector<Move> moreMoves;
  for (MoveType type : types) {
    moreMoves = possibleMoves(ann_network, type, start_nodes);
    res.insert(std::end(res), std::begin(moreMoves), std::end(moreMoves));
  }
  return res;
}

std::vector<Move> possibleMoves(AnnotatedNetwork &ann_network,
                                std::vector<MoveType> types, int min_radius,
                                int max_radius) {
  std::vector<Move> res;
  bool rspr1Present = (std::find(types.begin(), types.end(),
                                 MoveType::RSPR1Move) != types.end());
  bool deltaPlusPresent = (std::find(types.begin(), types.end(),
                                     MoveType::DeltaPlusMove) != types.end());
  std::vector<Move> moreMoves;
  for (MoveType type : types) {
    moreMoves = possibleMoves(ann_network, type, rspr1Present, deltaPlusPresent,
                              min_radius, max_radius);
    res.insert(std::end(res), std::begin(moreMoves), std::end(moreMoves));
  }
  return res;
}

Move getRandomMove(AnnotatedNetwork &ann_network,
                   std::vector<MoveType> typesBySpeed, int min_radius,
                   int max_radius) {
  std::vector<Move> allMoves =
      possibleMoves(ann_network, typesBySpeed, min_radius, max_radius);
  return allMoves[getRandomIndex(ann_network.rng, allMoves.size())];
}

void updateMoveBranchLengths(AnnotatedNetwork &ann_network,
                             std::vector<Move> &candidates) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    Move &move = candidates[i];
    if (isArcInsertion(move.moveType)) {
      move.arcInsertionData.a_b_len =
          get_edge_lengths(ann_network, move.arcInsertionData.ab_pmatrix_index);
      move.arcInsertionData.c_d_len =
          get_edge_lengths(ann_network, move.arcInsertionData.cd_pmatrix_index);
    } else if (isArcRemoval(move.moveType)) {
      move.arcRemovalData.a_u_len =
          get_edge_lengths(ann_network, move.arcRemovalData.au_pmatrix_index);
      move.arcRemovalData.u_b_len =
          get_edge_lengths(ann_network, move.arcRemovalData.ub_pmatrix_index);
      move.arcRemovalData.c_v_len =
          get_edge_lengths(ann_network, move.arcRemovalData.cv_pmatrix_index);
      move.arcRemovalData.v_d_len =
          get_edge_lengths(ann_network, move.arcRemovalData.vd_pmatrix_index);
      move.arcRemovalData.u_v_len =
          get_edge_lengths(ann_network, move.arcRemovalData.uv_pmatrix_index);
    } else if (isRSPR(move.moveType)) {
      size_t x_z_pmatrix_index =
          getEdgeTo(ann_network.network, move.rsprData.x_clv_index,
                    move.rsprData.z_clv_index)
              ->pmatrix_index;
      size_t z_y_pmatrix_index =
          getEdgeTo(ann_network.network, move.rsprData.z_clv_index,
                    move.rsprData.y_clv_index)
              ->pmatrix_index;
      move.rsprData.x_z_len = get_edge_lengths(ann_network, x_z_pmatrix_index);
      move.rsprData.z_y_len = get_edge_lengths(ann_network, z_y_pmatrix_index);
    }
  }
}

bool keepThisCandidate(AnnotatedNetwork &ann_network, const Move &move) {
  if (!checkSanity(ann_network, move)) {
    return false;
  }
  if (isArcInsertion(move.moveType)) {
    if (move.arcInsertionData.a_clv_index >= ann_network.network.num_nodes()) {
      return false;
    }
    if (move.arcInsertionData.b_clv_index >= ann_network.network.num_nodes()) {
      return false;
    }
    if (move.arcInsertionData.c_clv_index >= ann_network.network.num_nodes()) {
      return false;
    }
    if (move.arcInsertionData.d_clv_index >= ann_network.network.num_nodes()) {
      return false;
    }
  }
  return true;
}

void removeBadCandidates(AnnotatedNetwork &ann_network,
                         std::vector<Move> &candidates) {
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                  [&](const Move &move) {
                                    return !keepThisCandidate(ann_network,
                                                              move);
                                  }),
                   candidates.end());
  updateMoveBranchLengths(ann_network, candidates);
}

std::vector<Node *> gatherStartNodes(AnnotatedNetwork &ann_network,
                                     const Move &move) {
  std::vector<Node *> res;
  if (isArcInsertion(move.moveType)) {
    assert(ann_network.network
               .nodes_by_index[move.arcInsertionData.wanted_u_clv_index]);
    assert(ann_network.network
               .nodes_by_index[move.arcInsertionData.wanted_v_clv_index]);
    assert(move.arcInsertionData.wanted_u_clv_index <
           ann_network.network.num_nodes());
    assert(move.arcInsertionData.wanted_v_clv_index <
           ann_network.network.num_nodes());
    res.emplace_back(
        ann_network.network
            .nodes_by_index[move.arcInsertionData.wanted_u_clv_index]);
    res.emplace_back(
        ann_network.network
            .nodes_by_index[move.arcInsertionData.wanted_v_clv_index]);
  } else if (isRSPR(move.moveType)) {
    assert(ann_network.network.nodes_by_index[move.rsprData.z_clv_index]);
    assert(move.rsprData.z_clv_index < ann_network.network.num_nodes());
    res.emplace_back(
        ann_network.network.nodes_by_index[move.rsprData.z_clv_index]);
  } else if (move.moveType == MoveType::RNNIMove) {
    assert(ann_network.network.nodes_by_index[move.rnniData.v_clv_index]);
    assert(ann_network.network.nodes_by_index[move.rnniData.t_clv_index]);
    assert(move.rnniData.v_clv_index < ann_network.network.num_nodes());
    assert(move.rnniData.t_clv_index < ann_network.network.num_nodes());
    res.emplace_back(
        ann_network.network.nodes_by_index[move.rnniData.v_clv_index]);
    res.emplace_back(
        ann_network.network.nodes_by_index[move.rnniData.t_clv_index]);
  }
  return res;
}

void updateMoveBranchLengths(AnnotatedNetwork &ann_network, Move &move) {
  if (isArcInsertion(move.moveType)) {
    move.arcInsertionData.u_b_len = get_edge_lengths(
        ann_network, move.arcInsertionData.wanted_ub_pmatrix_index);
    move.arcInsertionData.u_v_len = get_edge_lengths(
        ann_network, move.arcInsertionData.wanted_uv_pmatrix_index);
    move.arcInsertionData.v_d_len = get_edge_lengths(
        ann_network, move.arcInsertionData.wanted_vd_pmatrix_index);
  } else if (isArcRemoval(move.moveType)) {
    move.arcRemovalData.a_b_len = get_edge_lengths(
        ann_network, move.arcRemovalData.wanted_ab_pmatrix_index);
    move.arcRemovalData.c_d_len = get_edge_lengths(
        ann_network, move.arcRemovalData.wanted_cd_pmatrix_index);
  }
}

void updateMovePmatrixIndex(Move &move, size_t old_pmatrix_index,
                            size_t new_pmatrix_index, bool undo) {
  if (isArcInsertion(move.moveType)) {
    updateMovePmatrixIndexArcInsertion(move, old_pmatrix_index,
                                       new_pmatrix_index, undo);
  } else if (isArcRemoval(move.moveType)) {
    updateMovePmatrixIndexArcRemoval(move, old_pmatrix_index, new_pmatrix_index,
                                     undo);
  } else if (move.moveType == MoveType::RNNIMove) {
    updateMovePmatrixIndexRNNI(move, old_pmatrix_index, new_pmatrix_index,
                               undo);
  } else if (isRSPR(move.moveType)) {
    updateMovePmatrixIndexRSPR(move, old_pmatrix_index, new_pmatrix_index,
                               undo);
  } else {
    throw std::runtime_error("unexpected move type");
  }
}

void updateMoveClvIndex(Move &move, size_t old_clv_index, size_t new_clv_index,
                        bool undo) {
  if (isArcInsertion(move.moveType)) {
    updateMoveClvIndexArcInsertion(move, old_clv_index, new_clv_index, undo);
  } else if (isArcRemoval(move.moveType)) {
    updateMoveClvIndexArcRemoval(move, old_clv_index, new_clv_index, undo);
  } else if (move.moveType == MoveType::RNNIMove) {
    updateMoveClvIndexRNNI(move, old_clv_index, new_clv_index, undo);
  } else if (isRSPR(move.moveType)) {
    updateMoveClvIndexRSPR(move, old_clv_index, new_clv_index, undo);
  } else {
    throw std::runtime_error("unexpected move type");
  }
}

}  // namespace netrax