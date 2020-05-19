/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Common.hpp"
#include <vector>
#include <random>

// TODO: Maybe put all moves into a class hierarchy?

namespace netrax {
struct AnnotatedNetwork;
// The moves correspond to the rNNI moves and rSPR moves from this paper: https://doi.org/10.1371/journal.pcbi.1005611

enum class MoveType {
    RNNIMove, RSPRMove, TailMove, HeadMove, RSPR1Move, ArcInsertionMove, ArcRemovalMove, DeltaPlusMove, DeltaMinusMove
};

enum class RNNIMoveType {
    ONE, ONE_STAR, TWO, TWO_STAR, THREE, THREE_STAR, FOUR
};

struct RNNIMove {
    size_t u_clv_index = 0;
    size_t v_clv_index = 0;
    size_t s_clv_index = 0;
    size_t t_clv_index = 0;
    RNNIMoveType type;
};

struct RSPRMove {
    size_t x_prime_clv_index = 0;
    size_t y_prime_clv_index = 0;
    size_t x_clv_index = 0;
    size_t y_clv_index = 0;
    size_t z_clv_index = 0;
};

struct ArcInsertionMove {
    size_t a_clv_index = 0;
    size_t b_clv_index = 0;
    size_t c_clv_index = 0;
    size_t d_clv_index = 0;

    double u_v_len = 1.0;
    double c_v_len = 1.0;
    double u_v_prob = 0.5;
    double c_v_prob = 0.5;
    double a_u_len = 1.0;
};

struct ArcRemovalMove {
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
    double a_u_length = 1.0;
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

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network, Node *v);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network);

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network, const Edge *edge);
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

}
