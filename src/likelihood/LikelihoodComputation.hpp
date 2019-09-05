/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

#include <pll/pll.h>

#include "../Network.hpp"

namespace netrax {

double computeLoglikelihood(const Network& network);

}
