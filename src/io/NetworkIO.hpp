/*
 * NetworkConverter.hpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "../graph/Network.hpp"
#include "../graph/AnnotatedNetwork.hpp"

#include <string>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

Network readNetworkFromString(const std::string &newick, int maxReticulations = -1);
Network readNetworkFromFile(const std::string &filename, int maxReticulations = -1);
std::string toExtendedNewick(AnnotatedNetwork &ann_network);
std::string toExtendedNewick(Network &network);
Network convertUtreeToNetwork(const pll_utree_t &utree, unsigned int maxReticulations);
void updateNetwork(AnnotatedNetwork &ann_network);

void writeNetwork(AnnotatedNetwork &ann_network, const std::string &filepath);

}
