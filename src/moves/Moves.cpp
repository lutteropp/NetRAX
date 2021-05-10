#include "Moves.hpp"

namespace netrax {

void performMove(AnnotatedNetwork &ann_network, GeneralMove *move) {
    assert(move);
    switch (move->moveType) {
        case MoveType::ArcInsertionMove:
            performMove(ann_network, *((ArcInsertionMove*) move));
            break;
        case MoveType::DeltaPlusMove:
            performMove(ann_network, *((ArcInsertionMove*) move));
            break;
        case MoveType::ArcRemovalMove:
            performMove(ann_network, *((ArcRemovalMove*) move));
            break;
        case MoveType::DeltaMinusMove:
            performMove(ann_network, *((ArcRemovalMove*) move));
            break;
        case MoveType::RNNIMove:
            performMove(ann_network, *((RNNIMove*) move));
            break;
        case MoveType::RSPRMove:
            performMove(ann_network, *((RSPRMove*) move));
            break;
        case MoveType::RSPR1Move:
            performMove(ann_network, *((RSPRMove*) move));
            break;
        case MoveType::HeadMove:
            performMove(ann_network, *((RSPRMove*) move));
            break;
        case MoveType::TailMove:
            performMove(ann_network, *((RSPRMove*) move));
            break;
        default:
            throw std::runtime_error("Invalid move type performMove: " + toString(move->moveType));
            break;
    }
}

void undoMove(AnnotatedNetwork &ann_network, GeneralMove *move) {
    assert(move);
    switch (move->moveType) {
        case MoveType::ArcInsertionMove:
            undoMove(ann_network, *((ArcInsertionMove*) move));
            break;
        case MoveType::DeltaPlusMove:
            undoMove(ann_network, *((ArcInsertionMove*) move));
            break;
        case MoveType::ArcRemovalMove:
            undoMove(ann_network, *((ArcRemovalMove*) move));
            break;
        case MoveType::DeltaMinusMove:
            undoMove(ann_network, *((ArcRemovalMove*) move));
            break;
        case MoveType::RNNIMove:
            undoMove(ann_network, *((RNNIMove*) move));
            break;
        case MoveType::RSPRMove:
            undoMove(ann_network, *((RSPRMove*) move));
            break;
        case MoveType::RSPR1Move:
            undoMove(ann_network, *((RSPRMove*) move));
            break;
        case MoveType::HeadMove:
            undoMove(ann_network, *((RSPRMove*) move));
            break;
        case MoveType::TailMove:
            undoMove(ann_network, *((RSPRMove*) move));
            break;
        default:
            throw std::runtime_error("Invalid move type undoMove: " + toString(move->moveType));
            break;
    }
}

std::string toString(GeneralMove *move) {
    assert(move);
    switch (move->moveType) {
        case MoveType::ArcInsertionMove:
            return toString(*((ArcInsertionMove*) move));
        case MoveType::DeltaPlusMove:
            return toString(*((ArcInsertionMove*) move));
        case MoveType::ArcRemovalMove:
            return toString(*((ArcRemovalMove*) move));
        case MoveType::DeltaMinusMove:
            return toString(*((ArcRemovalMove*) move));
        case MoveType::RNNIMove:
            return toString(*((RNNIMove*) move));
        case MoveType::RSPRMove:
            return toString(*((RSPRMove*) move));
        case MoveType::RSPR1Move:
            return toString(*((RSPRMove*) move));
        case MoveType::HeadMove:
            return toString(*((RSPRMove*) move));
        case MoveType::TailMove:
            return toString(*((RSPRMove*) move));
        default:
            throw std::runtime_error("Invalid move type toString: " + toString(move->moveType));
            break;
    }  
}

bool checkSanity(AnnotatedNetwork& ann_network, GeneralMove* move) {
    assert(move);
    switch (move->moveType) {
        case MoveType::ArcInsertionMove:
            return checkSanity(ann_network, *((ArcInsertionMove*) move));
        case MoveType::DeltaPlusMove:
            return checkSanity(ann_network, *((ArcInsertionMove*) move));
        case MoveType::ArcRemovalMove:
            return checkSanity(ann_network, *((ArcRemovalMove*) move));
        case MoveType::DeltaMinusMove:
            return checkSanity(ann_network, *((ArcRemovalMove*) move));
        case MoveType::RNNIMove:
            return checkSanity(ann_network, *((RNNIMove*) move));
        case MoveType::RSPRMove:
            return checkSanity(ann_network, *((RSPRMove*) move));
        case MoveType::RSPR1Move:
            return checkSanity(ann_network, *((RSPRMove*) move));
        case MoveType::HeadMove:
            return checkSanity(ann_network, *((RSPRMove*) move));
        case MoveType::TailMove:
            return checkSanity(ann_network, *((RSPRMove*) move));
        default:
            throw std::runtime_error("Invalid move type checkSanity: " + toString(move->moveType));
            break;
    }
}

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, GeneralMove* move) {
    assert(move);
    switch (move->moveType) {
        case MoveType::ArcInsertionMove:
            return brlenOptCandidates(ann_network, *((ArcInsertionMove*) move));
        case MoveType::DeltaPlusMove:
            return brlenOptCandidates(ann_network, *((ArcInsertionMove*) move));
        case MoveType::ArcRemovalMove:
            return brlenOptCandidates(ann_network, *((ArcRemovalMove*) move));
        case MoveType::DeltaMinusMove:
            return brlenOptCandidates(ann_network, *((ArcRemovalMove*) move));
        case MoveType::RNNIMove:
            return brlenOptCandidates(ann_network, *((RNNIMove*) move));
        case MoveType::RSPRMove:
            return brlenOptCandidates(ann_network, *((RSPRMove*) move));
        case MoveType::RSPR1Move:
            return brlenOptCandidates(ann_network, *((RSPRMove*) move));
        case MoveType::HeadMove:
            return brlenOptCandidates(ann_network, *((RSPRMove*) move));
        case MoveType::TailMove:
            return brlenOptCandidates(ann_network, *((RSPRMove*) move));
        default:
            throw std::runtime_error("Invalid move type brlenOptCandidates: " + toString(move->moveType));
            break;
    }
}

std::vector<GeneralMove*> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types) {
    std::vector<GeneralMove*> res;
    std::vector<ArcInsertionMove> insertionMoves;
    std::vector<ArcRemovalMove> removalMoves;
    std::vector<RNNIMove> rnniMoves;
    std::vector<RSPRMove> rsprMoves;
    for (MoveType type : types) {
        switch (type) {
            case MoveType::ArcInsertionMove:
                insertionMoves = possibleArcInsertionMoves(ann_network);
                for (size_t i = 0; i < insertionMoves.size(); ++i) {
                    res.emplace_back(new ArcInsertionMove(insertionMoves[i]));
                }
                break;
            case MoveType::DeltaPlusMove:
                insertionMoves = possibleDeltaPlusMoves(ann_network);
                for (size_t i = 0; i < insertionMoves.size(); ++i) {
                    res.emplace_back(new ArcInsertionMove(insertionMoves[i]));
                }
                break;
            case MoveType::ArcRemovalMove:
                removalMoves = possibleArcRemovalMoves(ann_network);
                for (size_t i = 0; i < removalMoves.size(); ++i) {
                    res.emplace_back(new ArcRemovalMove(removalMoves[i]));
                }
                break;
            case MoveType::DeltaMinusMove:
                removalMoves = possibleDeltaMinusMoves(ann_network);
                for (size_t i = 0; i < removalMoves.size(); ++i) {
                    res.emplace_back(new ArcRemovalMove(removalMoves[i]));
                }
                break;
            case MoveType::RNNIMove:
                rnniMoves = possibleRNNIMoves(ann_network);
                for (size_t i = 0; i < rnniMoves.size(); ++i) {
                    res.emplace_back(new RNNIMove(rnniMoves[i]));
                }
                break;
            case MoveType::RSPR1Move:
                rsprMoves = possibleRSPR1Moves(ann_network);
                for (size_t i = 0; i < rsprMoves.size(); ++i) {
                    res.emplace_back(new RSPRMove(rsprMoves[i]));
                }
                break;
            case MoveType::RSPRMove:
                rsprMoves = possibleRSPRMoves(ann_network);
                for (size_t i = 0; i < rsprMoves.size(); ++i) {
                    res.emplace_back(new RSPRMove(rsprMoves[i]));
                }
                break;
            case MoveType::HeadMove:
                rsprMoves = possibleHeadMoves(ann_network);
                for (size_t i = 0; i < rsprMoves.size(); ++i) {
                    res.emplace_back(new RSPRMove(rsprMoves[i]));
                }
                break;
            case MoveType::TailMove:
                rsprMoves = possibleTailMoves(ann_network);
                for (size_t i = 0; i < rsprMoves.size(); ++i) {
                    res.emplace_back(new RSPRMove(rsprMoves[i]));
                }
                break;
            default:
                throw std::runtime_error("Invalid move type possibleMoves: " + toString(type));
                break;
        }
    }
    return res;
}

GeneralMove* copyMove(GeneralMove* move) {
    assert(move);
    GeneralMove* copied_move = nullptr;
    switch (move->moveType) {
        case MoveType::ArcInsertionMove:
            copied_move = new ArcInsertionMove(*((ArcInsertionMove*) move));
            break;
        case MoveType::DeltaPlusMove:
            copied_move = new ArcInsertionMove(*((ArcInsertionMove*) move));
            break;
        case MoveType::ArcRemovalMove:
            copied_move = new ArcRemovalMove(*((ArcRemovalMove*) move));
            break;
        case MoveType::DeltaMinusMove:
            copied_move = new ArcRemovalMove(*((ArcRemovalMove*) move));
            break;
        case MoveType::RNNIMove:
            copied_move = new RNNIMove(*((RNNIMove*) move));
            break;
        case MoveType::RSPRMove:
            copied_move = new RSPRMove(*((RSPRMove*) move));
            break;
        case MoveType::RSPR1Move:
            copied_move = new RSPRMove(*((RSPRMove*) move));
            break;
        case MoveType::HeadMove:
            copied_move = new RSPRMove(*((RSPRMove*) move));
            break;
        case MoveType::TailMove:
            copied_move = new RSPRMove(*((RSPRMove*) move));
            break;
        default:
            throw std::runtime_error("Invalid move type brlenOptCandidates: " + toString(move->moveType));
            break;
    }
    return copied_move;
}

}