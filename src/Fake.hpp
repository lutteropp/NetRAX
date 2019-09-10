/*
 * Fake.hpp
 *
 *  Created on: Sep 9, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

extern "C"
{
#include <libpll/pll_tree.h>
#include <libpll/pll.h>
#include <libpll/pllmod_common.h>
}

#include <vector>
#include "PartitionInfo.hpp"

#include "Network.hpp"

namespace netrax {

struct NetworkParams {
	Network* network;
	pllmod_treeinfo_t* fake_treeinfo;
};

void destroy_fake_treeinfo(pllmod_treeinfo_t * treeinfo);
pllmod_treeinfo_t * create_fake_treeinfo(Network& network, unsigned int tips, unsigned int partitions, int brlen_linkage);

}
