/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "MoveType.hpp"

#include "ArcInsertionData.hpp"
#include "ArcRemovalData.hpp"
#include "RNNIData.hpp"
#include "RSPRData.hpp"

#include "../optimization/NetworkState.hpp"

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
// The moves correspond to the moves from this paper:
// https://doi.org/10.1371/journal.pcbi.1005611

struct Move {
  Move(MoveType type, size_t edge_orig_idx, size_t node_orig_idx)
      : moveType(type),
        edge_orig_idx(edge_orig_idx),
        node_orig_idx(node_orig_idx) {}
  MoveType moveType;
  size_t edge_orig_idx;
  size_t node_orig_idx;

  std::vector<std::pair<size_t, size_t>> remapped_clv_indices;
  std::vector<std::pair<size_t, size_t>> remapped_pmatrix_indices;
  std::vector<std::pair<size_t, size_t>> remapped_reticulation_indices;

  RNNIData rnniData;
  RSPRData rsprData;
  ArcInsertionData arcInsertionData;
  ArcRemovalData arcRemovalData;

  Move() = default;

  Move(Move &&rhs)
      : moveType{rhs.moveType},
        edge_orig_idx(rhs.edge_orig_idx),
        node_orig_idx(rhs.node_orig_idx),
        remapped_clv_indices{rhs.remapped_clv_indices},
        remapped_pmatrix_indices{rhs.remapped_pmatrix_indices},
        remapped_reticulation_indices{rhs.remapped_reticulation_indices},
        rnniData{rhs.rnniData},
        rsprData{rhs.rsprData},
        arcInsertionData{rhs.arcInsertionData},
        arcRemovalData{rhs.arcRemovalData} {}

  Move(const Move &rhs)
      : moveType{rhs.moveType},
        edge_orig_idx(rhs.edge_orig_idx),
        node_orig_idx(rhs.node_orig_idx),
        remapped_clv_indices{rhs.remapped_clv_indices},
        remapped_pmatrix_indices{rhs.remapped_pmatrix_indices},
        remapped_reticulation_indices{rhs.remapped_reticulation_indices},
        rnniData{rhs.rnniData},
        rsprData{rhs.rsprData},
        arcInsertionData{rhs.arcInsertionData},
        arcRemovalData{rhs.arcRemovalData} {}

  Move &operator=(Move &&rhs) {
    if (this != &rhs) {
      moveType = rhs.moveType;
      edge_orig_idx = rhs.edge_orig_idx;
      node_orig_idx = rhs.node_orig_idx;
      remapped_clv_indices = rhs.remapped_clv_indices;
      remapped_pmatrix_indices = rhs.remapped_pmatrix_indices;
      remapped_reticulation_indices = rhs.remapped_reticulation_indices;
      rnniData = rhs.rnniData;
      rsprData = rhs.rsprData;
      arcInsertionData = rhs.arcInsertionData;
      arcRemovalData = rhs.arcRemovalData;
    }
    return *this;
  }

  Move &operator=(const Move &rhs) {
    if (this != &rhs) {
      moveType = rhs.moveType;
      edge_orig_idx = rhs.edge_orig_idx;
      node_orig_idx = rhs.node_orig_idx;
      remapped_clv_indices = rhs.remapped_clv_indices;
      remapped_pmatrix_indices = rhs.remapped_pmatrix_indices;
      remapped_reticulation_indices = rhs.remapped_reticulation_indices;
      rnniData = rhs.rnniData;
      rsprData = rhs.rsprData;
      arcInsertionData = rhs.arcInsertionData;
      arcRemovalData = rhs.arcRemovalData;
    }
    return *this;
  }

  bool operator==(const Move &rhs) const {
    return ((this->moveType == rhs.moveType) &&
            (this->edge_orig_idx == rhs.edge_orig_idx) &&
            (this->node_orig_idx == rhs.node_orig_idx) &&
            (this->remapped_clv_indices == rhs.remapped_clv_indices) &&
            (this->remapped_pmatrix_indices == rhs.remapped_pmatrix_indices) &&
            (this->remapped_reticulation_indices ==
             rhs.remapped_reticulation_indices) &&
            (this->rnniData == rhs.rnniData) &&
            (this->rsprData == rhs.rsprData) &&
            (this->arcInsertionData == rhs.arcInsertionData) &&
            (this->arcRemovalData == rhs.arcRemovalData));
  }
};

Move randomMove(AnnotatedNetwork &ann_network, MoveType type);
Move getRandomMove(AnnotatedNetwork& ann_network, std::vector<MoveType> typesBySpeed, int min_radius, int max_radius);
void performMove(AnnotatedNetwork &ann_network, Move &move);
void undoMove(AnnotatedNetwork &ann_network, Move &move);
std::string toString(const Move &move);
bool checkSanity(AnnotatedNetwork &ann_network, const Move &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network,
                                              const Move &move);

std::vector<Move> possibleMoves(
    AnnotatedNetwork &ann_network, MoveType type,
    std::vector<Edge *> start_edges, bool rspr1_present = false,
    bool delta_plus_present = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(
    AnnotatedNetwork &ann_network, MoveType type,
    std::vector<Node *> start_nodes, bool rspr1_present = false,
    bool delta_plus_present = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(
    AnnotatedNetwork &ann_network, MoveType type, bool rspr1_present = false,
    bool delta_plus_present = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMoves(
    AnnotatedNetwork &ann_network, std::vector<MoveType> types,
    std::vector<Edge *> start_edges, bool rspr1_present = false,
    bool delta_plus_present = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(
    AnnotatedNetwork &ann_network, std::vector<MoveType> types,
    std::vector<Node *> start_nodes, bool rspr1_present = false,
    bool delta_plus_present = false, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(
    AnnotatedNetwork &ann_network, std::vector<MoveType> types,
    int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

void removeBadCandidates(AnnotatedNetwork &ann_network,
                         std::vector<Move> &candidates);
std::vector<Node *> gatherStartNodes(AnnotatedNetwork &ann_network,
                                     const Move &move);

void updateMoveBranchLengths(AnnotatedNetwork &ann_network, Move &move);
void updateMovePmatrixIndex(Move &move, size_t old_pmatrix_index,
                            size_t new_pmatrix_index, bool undo);
void updateMoveClvIndex(Move &move, size_t old_clv_index, size_t new_clv_index,
                        bool undo);

}  // namespace netrax
