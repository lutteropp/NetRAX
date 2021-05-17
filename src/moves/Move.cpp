#include "Move.hpp"

#include "ArcRemoval.hpp"
#include "ArcInsertion.hpp"
#include "RNNI.hpp"
#include "RSPR.hpp"

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
            throw std::runtime_error("Invalid move type randomMove: " + toString(type));
    }  
}

void performMove(AnnotatedNetwork &ann_network, Move& move) {
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
            throw std::runtime_error("Invalid move type performMove: " + toString(move.moveType));
            break;
    }
}

void undoMove(AnnotatedNetwork &ann_network, Move& move) {
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
            throw std::runtime_error("Invalid move type undoMove: " + toString(move.moveType));
            break;
    }
}

std::string toString(const Move& move) {
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
            throw std::runtime_error("Invalid move type toString: " + toString(move.moveType));
    }
}

bool checkSanity(AnnotatedNetwork& ann_network, const Move& move) {
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
            throw std::runtime_error("Invalid move type checkSanity: " + toString(move.moveType));
    }
}

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, const Move& move) {
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
            throw std::runtime_error("Invalid move type brlenOptCandidates: " + toString(move.moveType));
    }
}

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, MoveType type, bool rspr1_present, bool delta_plus_present, int min_radius, int max_radius) {
    switch (type) {
        case MoveType::ArcInsertionMove:
            return possibleMovesArcInsertion(ann_network, delta_plus_present, min_radius, max_radius);
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
            return possibleMovesRSPR(ann_network, rspr1_present, min_radius, max_radius);
        case MoveType::HeadMove:
            return possibleMovesHead(ann_network, rspr1_present, min_radius, max_radius);
        case MoveType::TailMove:
            return possibleMovesTail(ann_network, rspr1_present, min_radius, max_radius);
        default:
            throw std::runtime_error("Invalid move type possibleMoves: " + toString(type));
    }
}

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, MoveType type, std::vector<Edge*> start_edges, bool rspr1_present, bool delta_plus_present, int min_radius, int max_radius) {
    switch (type) {
        case MoveType::ArcInsertionMove:
            return possibleMovesArcInsertion(ann_network, start_edges, delta_plus_present, min_radius, max_radius);
        case MoveType::DeltaPlusMove:
            return possibleMovesDeltaPlus(ann_network, start_edges, min_radius, max_radius);
        case MoveType::ArcRemovalMove:
            return possibleMovesArcRemoval(ann_network, start_edges);
        case MoveType::DeltaMinusMove:
            return possibleMovesDeltaMinus(ann_network, start_edges);
        case MoveType::RNNIMove:
            return possibleMovesRNNI(ann_network, start_edges);
        case MoveType::RSPR1Move:
            return possibleMovesRSPR1(ann_network, start_edges, min_radius, max_radius);
        case MoveType::RSPRMove:
            return possibleMovesRSPR(ann_network, start_edges, rspr1_present, min_radius, max_radius);
        case MoveType::HeadMove:
            return possibleMovesHead(ann_network, start_edges, rspr1_present, min_radius, max_radius);
        case MoveType::TailMove:
            return possibleMovesTail(ann_network, start_edges, rspr1_present, min_radius, max_radius);
        default:
            throw std::runtime_error("Invalid move type possibleMoves: " + toString(type));
    }
}

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, MoveType type, std::vector<Node*> start_nodes, bool rspr1_present, bool delta_plus_present, int min_radius, int max_radius) {
    switch (type) {
        case MoveType::ArcInsertionMove:
            return possibleMovesArcInsertion(ann_network, start_nodes, delta_plus_present, min_radius, max_radius);
        case MoveType::DeltaPlusMove:
            return possibleMovesDeltaPlus(ann_network, start_nodes, min_radius, max_radius);
        case MoveType::ArcRemovalMove:
            return possibleMovesArcRemoval(ann_network, start_nodes);
        case MoveType::DeltaMinusMove:
            return possibleMovesDeltaMinus(ann_network, start_nodes);
        case MoveType::RNNIMove:
            return possibleMovesRNNI(ann_network, start_nodes);
        case MoveType::RSPR1Move:
            return possibleMovesRSPR1(ann_network, start_nodes, min_radius, max_radius);
        case MoveType::RSPRMove:
            return possibleMovesRSPR(ann_network, start_nodes, rspr1_present, min_radius, max_radius);
        case MoveType::HeadMove:
            return possibleMovesHead(ann_network, start_nodes, rspr1_present, min_radius, max_radius);
        case MoveType::TailMove:
            return possibleMovesTail(ann_network, start_nodes, rspr1_present, min_radius, max_radius);
        default:
            throw std::runtime_error("Invalid move type possibleMoves: " + toString(type));
    }
}

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types, std::vector<Edge*> start_edges) {
    std::vector<Move> res;
    std::vector<Move> moreMoves;
    for (MoveType type : types) {
        moreMoves = possibleMoves(ann_network, type, start_edges);
        res.insert(std::end(res), std::begin(moreMoves), std::end(moreMoves));
    }
    return res;
}

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types, std::vector<Node*> start_nodes) {
    std::vector<Move> res;
    std::vector<Move> moreMoves;
    for (MoveType type : types) {
        moreMoves = possibleMoves(ann_network, type, start_nodes);
        res.insert(std::end(res), std::begin(moreMoves), std::end(moreMoves));
    }
    return res;
}

std::vector<Move> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types) {
    std::vector<Move> res;
    std::vector<Move> moreMoves;
    for (MoveType type : types) {
        moreMoves = possibleMoves(ann_network, type);
        res.insert(std::end(res), std::begin(moreMoves), std::end(moreMoves));
    }
    return res;
}

}