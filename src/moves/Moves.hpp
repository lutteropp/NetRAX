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

// TODO: Maybe put all moves into a class hierarchy?

#include "RNNIMove.hpp"
#include "RSPRMove.hpp"
#include "ArcInsertionMove.hpp"
#include "ArcRemovalMove.hpp"

namespace netrax {
// The moves correspond to the moves from this paper: https://doi.org/10.1371/journal.pcbi.1005611

void performMove(AnnotatedNetwork &ann_network, GeneralMove *move);
void undoMove(AnnotatedNetwork &ann_network, GeneralMove *move);
std::string toString(GeneralMove *move);
bool checkSanity(AnnotatedNetwork& ann_network, GeneralMove* move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, GeneralMove* move);
std::vector<GeneralMove*> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types);
GeneralMove* copyMove(GeneralMove* move);

}
