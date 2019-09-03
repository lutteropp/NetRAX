/*
 * NetworkConverter.hpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../Network.hpp"

extern "C" {
#include "lowlevel_parsing.h"
}

namespace netrax {

Network readNetworkFromString(const std::string& newick);
Network readNetworkFromFile(const std::string& filename);
std::string toExtendedNewick(const Network& network);

}
