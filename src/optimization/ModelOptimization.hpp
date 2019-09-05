/*
 * ModelOptimization.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once


#include "../Network.hpp"
#include "../PartitionInfo.hpp"
#include <vector>

namespace netrax {
	void optimizeModel(Network& network, std::vector<PartitionInfo>& partitions, double lh_epsilon);
}
