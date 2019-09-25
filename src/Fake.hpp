/*
 * Fake.hpp
 *
 *  Created on: Sep 9, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

extern "C" {
#include <libpll/pll_tree.h>
#include <libpll/pll.h>
#include <libpll/pllmod_common.h>
}

#include <raxml-ng/TreeInfo.hpp>
#include <raxml-ng/main.hpp>

#include <vector>

#include "Network.hpp"

namespace netrax {

struct NetworkParams {
	Network* network;
	pllmod_treeinfo_t* fake_treeinfo;
};

void destroy_fake_treeinfo(pllmod_treeinfo_t * treeinfo);
//pllmod_treeinfo_t * create_fake_treeinfo(Network& network, unsigned int tips, unsigned int partitions, int brlen_linkage);

TreeInfo create_fake_raxml_treeinfo(Network& network, const Options &opts, const std::vector<doubleVector>& partition_brlens,
		const PartitionedMSA& parted_msa, const IDVector& tip_msa_idmap, const PartitionAssignment& part_assign);
TreeInfo create_fake_raxml_treeinfo(Network& network, const Options &opts, const std::vector<doubleVector>& partition_brlens,
		const PartitionedMSA& parted_msa, const IDVector& tip_msa_idmap, const PartitionAssignment& part_assign,
		const std::vector<uintVector>& site_weights);
TreeInfo create_fake_raxml_treeinfo(Network& network, const Options &opts, const PartitionedMSA& parted_msa, const IDVector& tip_msa_idmap,
		const PartitionAssignment& part_assign);
TreeInfo create_fake_raxml_treeinfo(Network& network, const Options &opts, const PartitionedMSA& parted_msa, const IDVector& tip_msa_idmap,
		const PartitionAssignment& part_assign, const std::vector<uintVector>& site_weights);

RaxmlInstance createStandardRaxmlInstance(const std::string& treePath, const std::string& msaPath, bool useRepeats = false);

TreeInfo createStandardRaxmlTreeinfo(const std::string& treePath, const std::string& msaPath, bool useRepeats = false);

TreeInfo createFakeRaxmlTreeinfo(Network& network, const std::string& networkPath, const std::string& msaPath, bool useRepeats = false);

}
