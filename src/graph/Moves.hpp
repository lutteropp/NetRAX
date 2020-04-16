/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Common.hpp"

namespace netrax {

	enum class RNNIMove {
		A1, A2, B1, B2, C1, C2, D
	};

	std::vector<RNNIMove> possibleRNNIMoves(const Network& network, const Edge& edge);
	void performRNNIMove(Network& network, RNNIMove& move);
	void undoRNNIMove(Network&network, RNNIMove& move, Edge& edge);

}
