#pragma once

#include <stdexcept>
#include <string>

namespace netrax {
    enum class MoveType {
        INVALID,
        RNNIMove,
        RSPRMove,
        TailMove,
        HeadMove,
        RSPR1Move,
        ArcInsertionMove,
        ArcRemovalMove,
        DeltaPlusMove,
        DeltaMinusMove
    };

    inline std::string toString(MoveType type) {
        switch (type) {
        case MoveType::ArcInsertionMove:
            return "ArcInsertionMove";
        case MoveType::ArcRemovalMove:
            return "ArcRemovalMove";
        case MoveType::DeltaMinusMove:
            return "DeltaMinusMove";
        case MoveType::DeltaPlusMove:
            return "DeltaPlusMove";
        case MoveType::HeadMove:
            return "HeadMove";
        case MoveType::RNNIMove:
            return "RNNIMove";
        case MoveType::RSPR1Move:
            return "RSPR1Move";
        case MoveType::RSPRMove:
            return "RSPRMove";
        case MoveType::TailMove:
            return "TailMove";
        default:
            throw std::runtime_error("Invalid move type");
        }
    }

    inline bool isArcInsertion(const MoveType& type) {
        return (type == MoveType::ArcInsertionMove || type == MoveType::DeltaPlusMove);
    }

    inline bool isArcRemoval(const MoveType& type) {
        return (type == MoveType::ArcRemovalMove || type == MoveType::DeltaMinusMove);
    }

    inline bool isRSPR(const MoveType& type) {
        return (type == MoveType::RSPRMove || type == MoveType::RSPR1Move || type == MoveType::HeadMove || type == MoveType::TailMove);
    }

    inline bool isComplexityChangingMove(const MoveType& moveType) {
        return isArcInsertion(moveType) || isArcRemoval(moveType);
    }

}