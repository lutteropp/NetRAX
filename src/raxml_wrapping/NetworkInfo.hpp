/*
 * NetworkInfo.hpp
 *
 *  Created on: Sep 13, 2019
 *      Author: Sarah Lutteropp
 */

#include <raxml-ng/TreeInfo.hpp>

#pragma once

namespace netrax {

class NetworkInfo: public TreeInfo {
	NetworkInfo(const Options &opts, const Tree& tree, const PartitionedMSA& parted_msa, const IDVector& tip_msa_idmap,
			const PartitionAssignment& part_assign);
	NetworkInfo(const Options &opts, const Tree& tree, const PartitionedMSA& parted_msa, const IDVector& tip_msa_idmap,
			const PartitionAssignment& part_assign, const std::vector<uintVector>& site_weights);
	virtual
	~NetworkInfo();
};

}
