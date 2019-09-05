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
#include "../PartitionInfo.hpp"

namespace netrax {

void updateProbMatrices(Network& network, std::vector<PartitionInfo>& partitions, bool updateAll);
double computeLoglikelihood(Network& network, std::vector<PartitionInfo>& partitions);

}
