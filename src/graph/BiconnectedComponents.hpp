/*
 * BiconnectedComponents.hpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#include "Network.hpp"

#include <vector>

namespace netrax {
	std::vector<unsigned int> partitionNetworkIntoBlobs(const Network& network);
}
