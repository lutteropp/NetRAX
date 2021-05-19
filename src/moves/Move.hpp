/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>

#include "MoveType.hpp"

#include "RNNIData.hpp"
#include "RSPRData.hpp"
#include "ArcInsertionData.hpp"
#include "ArcRemovalData.hpp"

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
// The moves correspond to the moves from this paper: https://doi.org/10.1371/journal.pcbi.1005611

struct Move {
    Move(MoveType type, size_t edge_orig_idx, size_t node_orig_idx) :
            moveType(type), edge_orig_idx(edge_orig_idx), node_orig_idx(node_orig_idx) {
    }
    MoveType moveType;
    size_t edge_orig_idx;
    size_t node_orig_idx;

    RNNIData rnniData;
    RSPRData rsprData;
    ArcInsertionData arcInsertionData;
    ArcRemovalData arcRemovalData;

    Move() = default;

    Move(Move&& rhs) : moveType{rhs.moveType}, edge_orig_idx(rhs.edge_orig_idx), node_orig_idx(rhs.node_orig_idx), rnniData{rhs.rnniData}, rsprData{rhs.rsprData}, arcInsertionData{rhs.arcInsertionData}, arcRemovalData{rhs.arcRemovalData} {}

    Move(const Move& rhs) : moveType{rhs.moveType}, edge_orig_idx(rhs.edge_orig_idx), node_orig_idx(rhs.node_orig_idx), rnniData{rhs.rnniData}, rsprData{rhs.rsprData}, arcInsertionData{rhs.arcInsertionData}, arcRemovalData{rhs.arcRemovalData} {}

    Move& operator =(Move&& rhs)
    {
        if (this != &rhs)
        {
            moveType = rhs.moveType;
            edge_orig_idx = rhs.edge_orig_idx;
            node_orig_idx = rhs.node_orig_idx;
            rnniData = rhs.rnniData;
            rsprData = rhs.rsprData;
            arcInsertionData = rhs.arcInsertionData;
            arcRemovalData = rhs.arcRemovalData;
        }
        return *this;
    }

    Move& operator =(const Move& rhs)
    {
        if (this != &rhs)
        {
            moveType = rhs.moveType;
            edge_orig_idx = rhs.edge_orig_idx;
            node_orig_idx = rhs.node_orig_idx;
            rnniData = rhs.rnniData;
            rsprData = rhs.rsprData;
            arcInsertionData = rhs.arcInsertionData;
            arcRemovalData = rhs.arcRemovalData;
        }
        return *this;
    }

    bool operator==(const Move& rhs) const { 
        return(
            (this->moveType == rhs.moveType)
            && (this->edge_orig_idx == rhs.edge_orig_idx)
            && (this->node_orig_idx == rhs.node_orig_idx)
            && (this->rnniData == rhs.rnniData)
            && (this->rsprData == rhs.rsprData)
            && (this->arcInsertionData == rhs.arcInsertionData)
            && (this->arcRemovalData == rhs.arcRemovalData)
        );
    }

};

Move randomMove(AnnotatedNetwork &ann_network, MoveType type);
void performMove(AnnotatedNetwork &ann_network, Move& move);
void undoMove(AnnotatedNetwork &ann_network, Move& move);
std::string toString(const Move& move);
bool checkSanity(AnnotatedNetwork& ann_network, const Move& move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, const Move& move);

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, MoveType type, std::vector<Edge*> start_edges, bool rspr1_present = false, bool delta_plus_present = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, MoveType type, std::vector<Node*> start_nodes, bool rspr1_present = false, bool delta_plus_present = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, MoveType type, bool rspr1_present = false, bool delta_plus_present = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types, std::vector<Edge*> start_edges, bool rspr1_present = false, bool delta_plus_present = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types, std::vector<Node*> start_nodes, bool rspr1_present = false, bool delta_plus_present = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());
std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types, bool rspr1_present = false, bool delta_plus_present = false, int min_radius = 0, int max_radius = std::numeric_limits<int>::max());

void removeBadCandidates(AnnotatedNetwork& ann_network, std::vector<Move>& candidates);
std::vector<Node*> gatherStartNodes(AnnotatedNetwork& ann_network, Move move);

void updateMoveBranchLengths(AnnotatedNetwork& ann_network, Move& move);

}
