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

// TODO: Maybe put all moves into a class hierarchy?

namespace netrax {
struct AnnotatedNetwork;
struct Edge;
struct Node;
// The moves correspond to the rNNI moves and rSPR moves from this paper: https://doi.org/10.1371/journal.pcbi.1005611

enum class MoveType {
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

struct GeneralMove {
    GeneralMove(MoveType type) :
            moveType(type) {
    }
    MoveType moveType;
};

enum class RNNIMoveType {
    ONE, ONE_STAR, TWO, TWO_STAR, THREE, THREE_STAR, FOUR
};

struct RNNIMove: public GeneralMove {
    RNNIMove() :
            GeneralMove(MoveType::RNNIMove) {
    }

    size_t u_clv_index = 0;
    size_t v_clv_index = 0;
    size_t s_clv_index = 0;
    size_t t_clv_index = 0;
    RNNIMoveType type = RNNIMoveType::ONE;
};

struct RSPRMove: public GeneralMove {
    RSPRMove() :
            GeneralMove(MoveType::RSPRMove) {
    }
    size_t x_prime_clv_index = 0;
    size_t y_prime_clv_index = 0;
    size_t x_clv_index = 0;
    size_t y_clv_index = 0;
    size_t z_clv_index = 0;

    double x_z_len = 0;
    double z_y_len = 0;
    double x_prime_y_prime_len = 0;
    double x_z_prob = 0.5;
    double z_y_prob = 0.5;
    double x_prime_y_prime_prob = 0.5;
};

struct ArcInsertionMove: public GeneralMove {
    ArcInsertionMove() :
            GeneralMove(MoveType::ArcInsertionMove) {
    }
    size_t a_clv_index = 0;
    size_t b_clv_index = 0;
    size_t c_clv_index = 0;
    size_t d_clv_index = 0;

    double u_v_len = 1.0;
    double c_v_len = 1.0;
    double u_v_prob = 0.5;
    double c_v_prob = 0.5;
    double a_u_len = 1.0;

    size_t wanted_u_clv_index = std::numeric_limits<size_t>::max();
    size_t wanted_v_clv_index = std::numeric_limits<size_t>::max();
    size_t wanted_au_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_ub_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_cv_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_vd_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_uv_pmatrix_index = std::numeric_limits<size_t>::max();

    size_t ab_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t cd_pmatrix_index = std::numeric_limits<size_t>::max();
};

struct ArcRemovalMove: public GeneralMove {
    ArcRemovalMove() :
            GeneralMove(MoveType::ArcRemovalMove) {
    }
    size_t a_clv_index = 0;
    size_t b_clv_index = 0;
    size_t c_clv_index = 0;
    size_t d_clv_index = 0;
    size_t u_clv_index = 0;
    size_t v_clv_index = 0;

    double u_v_len = 0.0;
    double c_v_len = 0.0;
    double u_v_prob = 0.5;
    double c_v_prob = 0.5;
    double a_u_len = 1.0;

    size_t au_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t ub_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t cv_pmatric_index = std::numeric_limits<size_t>::max();
    size_t vd_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t uv_pmatrix_index = std::numeric_limits<size_t>::max();

    size_t wanted_ab_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_cd_pmatrix_index = std::numeric_limits<size_t>::max();
};

std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network);
std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network);

std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network);
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network);

void performMove(AnnotatedNetwork &ann_network, RNNIMove &move);
void performMove(AnnotatedNetwork &ann_network, RSPRMove &move);
void undoMove(AnnotatedNetwork &ann_network, RNNIMove &move);
void undoMove(AnnotatedNetwork &ann_network, RSPRMove &move);

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const Edge *edge);
std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network);

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network,
        const Edge *edge);
std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network);
std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network, Node *v);
std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network);

void performMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);
void performMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);
void undoMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);
void undoMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);

std::string toString(RNNIMove &move);
std::string toString(RSPRMove &move);
std::string toString(ArcInsertionMove &move);
std::string toString(ArcRemovalMove &move);
std::string toString(MoveType &type);

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RNNIMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RSPRMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, ArcRemovalMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RNNIMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RSPRMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcRemovalMove &move);

void performMove(AnnotatedNetwork &ann_network, GeneralMove *move);
void undoMove(AnnotatedNetwork &ann_network, GeneralMove *move);
std::string toString(GeneralMove *move);

}