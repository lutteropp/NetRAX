/*
 * NetworkConverter.hpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../graph/Common.hpp"

#include "RootedNetworkParser.hpp"

namespace netrax {

Network convertNetwork(const RootedNetwork& rnetwork);
Network readNetworkFromString(const std::string& newick);
Network readNetworkFromFile(const std::string& filename);
std::string toExtendedNewick(const Network& network);

}
