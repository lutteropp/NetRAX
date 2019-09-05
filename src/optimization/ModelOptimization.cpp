/*
 * ModelOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "ModelOptimization.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../Network.hpp"
#include <raxml-ng/constants.hpp>

namespace netrax {

void optimize_params_all(double lh_epsilon) {
	// TODO: Reimplement networkinfo.optimize_params_all(lh_epsilon);
}

double optimizeModel(Network& network, std::vector<PartitionInfo>& partitions, double lh_epsilon) {
	double new_loglh = computeLoglikelihood(network, partitions);
	double cur_loglh;
	do {
		cur_loglh = new_loglh;
		optimize_params_all(lh_epsilon);
		new_loglh = computeLoglikelihood(network, partitions);
		assert(cur_loglh - new_loglh < -new_loglh * RAXML_DOUBLE_TOLERANCE);
	} while (new_loglh - cur_loglh > lh_epsilon);
	return new_loglh;
}
}
