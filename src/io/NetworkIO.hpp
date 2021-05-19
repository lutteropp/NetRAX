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

Network readNetworkFromString(const std::string &newick, const NetraxOptions& options, int maxReticulations = -1);
Network readNetworkFromFile(const std::string &filename,const NetraxOptions& options,  int maxReticulations = -1);
std::string toExtendedNewick(AnnotatedNetwork &ann_network);
std::string toExtendedNewick(Network &network);
Network convertUtreeToNetwork(const pll_utree_t &utree, NetraxOptions& options, unsigned int maxReticulations);
void collect_average_branches(AnnotatedNetwork &ann_network);
void updateNetwork(AnnotatedNetwork &ann_network);

void writeNetwork(AnnotatedNetwork &ann_network, const std::string &filepath);

}
