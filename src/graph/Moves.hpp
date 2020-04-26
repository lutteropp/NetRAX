/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Common.hpp"
#include <vector>

// TODO: Maybe put all moves into a class hierarchy?

namespace netrax {
struct AnnotatedNetwork;
// The moves correspond to the rNNI moves and rSPR moves from this paper: https://doi.org/10.1371/journal.pcbi.1005611

enum class RNNIMoveType {
    ONE, ONE_STAR, TWO, TWO_STAR, THREE, THREE_STAR, FOUR
};

struct RNNIMove {
    Node *u = nullptr;
    Node *v = nullptr;
    Node *s = nullptr;
    Node *t = nullptr;
    RNNIMoveType type;
};

struct RSPRMove {
    Node *x_prime = nullptr;
    Node *y_prime = nullptr;
    Node *x = nullptr;
    Node *y = nullptr;
    Node *z = nullptr;
};

struct ArcInsertionMove {
    Node *a = nullptr;
    Node *b = nullptr;
    Node *c = nullptr;
    Node *d = nullptr;
};

struct ArcRemovalMove {
    Node *a = nullptr;
    Node *b = nullptr;
    Node *c = nullptr;
    Node *d = nullptr;
    Node *u = nullptr;
    Node *v = nullptr;
};

std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network, const Edge &edge);
std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge &edge);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, const Edge &edge);
std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network);
std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network);
void performMove(AnnotatedNetwork &ann_network, RNNIMove &move);
void performMove(AnnotatedNetwork &ann_network, RSPRMove &move);
void undoMove(AnnotatedNetwork &ann_network, RNNIMove &move);
void undoMove(AnnotatedNetwork &ann_network, RSPRMove &move);

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, const Edge& edge);
std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network, Node *v);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network);
void performMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);
void performMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);
void undoMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);
void undoMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);

std::string toString(RNNIMove &move);
std::string toString(RSPRMove &move);
std::string toString(ArcInsertionMove &move);
std::string toString(ArcRemovalMove &move);

}
