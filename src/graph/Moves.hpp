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
		A1, A2, A1_alt, A2_alt, B1, B2, C1, C2, D, D_alt
		/*
		 * The moves correspond to the moves in this paper: https://doi.org/10.1371/journal.pcbi.1005611
		 * A1 is for 1, A2 is for 1*, B1 is for 2, B2 is for 2*, C1 is for 3, C2 is for 3*, D is for 4.
		 * The alt moves are for handling that the root node has 3 children in our network data structure.
		 */
	};

	std::vector<RNNIMove> possibleRNNIMoves(const Network& network, const Edge& edge);
	void performMove(Network& network, RNNIMove& move, Edge& edge);
	void undoMove(Network&network, RNNIMove& move, Edge& edge);

}
