/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental = 1,
                            int update_pmatrices = 1);

}
