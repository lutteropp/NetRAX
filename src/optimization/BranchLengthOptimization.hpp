/*
 * BranchLengthOptimization.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../Network.hpp"

namespace netrax {
	double optimize_branches(Network& network, std::vector<PartitionInfo>& partitions, double lh_epsilon, double brlen_smooth_factor);
}
