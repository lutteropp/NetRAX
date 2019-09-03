/*
 * NetworkConverter.hpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "lowlevel_parsing.h"
#include "../Network.hpp"

namespace netrax {

Network readNetworkFromString(const std::string& newick);
Network readNetworkFromFile(const std::string& filename);
std::string toExtendedNewick(const Network& network);

}
