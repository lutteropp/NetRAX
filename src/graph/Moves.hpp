/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Common.hpp"

namespace netrax {
    bool hasPath(const Network& network, const Node* from, const Node* to);
}