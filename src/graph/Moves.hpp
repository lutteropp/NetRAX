/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Common.hpp"

namespace netrax {
    // The moves correspond to the rNNI moves in this paper: https://doi.org/10.1371/journal.pcbi.1005611
	struct RNNIMove {
		Node* u = nullptr;
		Node* v = nullptr;
		Node* s = nullptr;
		Node* t = nullptr;
		bool swapUVDirection = false;
	};

	std::vector<RNNIMove> possibleRNNIMoves(const Network& network, const Edge& edge);
	void performMove(Network& network, RNNIMove& move);
	void undoMove(Network&network, RNNIMove& move);

}
