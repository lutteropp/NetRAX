/*
 * BiconnectedComponents.hpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#include "Network.hpp"

#include <vector>

namespace netrax {
	std::vector<unsigned int> partitionNetworkNodesIntoBlobs(const Network& network);
	std::vector<unsigned int> partitionNetworkEdgesIntoBlobs(const Network& network);
}
