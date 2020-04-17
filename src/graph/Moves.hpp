/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Common.hpp"
#include <vector>

namespace netrax {
struct Network;
// The moves correspond to the rNNI moves in this paper: https://doi.org/10.1371/journal.pcbi.1005611

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
    Node* y_prime = nullptr;
    Node *x = nullptr;
    Node *y = nullptr;
    Node *z = nullptr;
};

std::vector<RNNIMove> possibleRNNIMoves(Network &network, const Edge &edge);
std::vector<RSPRMove> possibleRSPRMoves(Network &network, const Edge &edge);
void performMove(Network &network, RNNIMove &move);
void performMove(Network &, RSPRMove& move);
void undoMove(Network &network, RNNIMove &move);
void undoMove(Network &, RSPRMove &move);

}
