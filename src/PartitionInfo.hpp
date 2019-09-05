/*
 * PartitionInfo.hpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <libpll/pll.h>
#include <vector>
#include <stdexcept>

namespace netrax {

struct PartitionInfo {
	void init(const Network& network);

	pll_partition_t pll_partition;
	std::vector<unsigned int> param_indices; // ??? "index of the model of whose frequency vector is to be used" ??? length is equal to pll_partition.rate_cats
	std::vector<double> branch_lengths;
	std::vector<bool> pmatrix_valid; // this is for the edges
	std::vector<bool> clv_valid; // this is for the links
	double brlen_scaler;
};

void PartitionInfo::init(const Network& network) {
	throw std::runtime_error("Not implemented yet");
}

}
