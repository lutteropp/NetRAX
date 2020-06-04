/*
 * OldLikelihoodComputation.hpp
 *
 *  Created on: Jun 4, 2020
 *      Author: sarah
 */

#pragma once

#include <stddef.h>
#include <vector>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "src/graph/Common.hpp"
#include "src/graph/AnnotatedNetwork.hpp"
#include "src/RaxmlWrapper.hpp"
#include <raxml-ng/TreeInfo.hpp>

namespace netrax {
namespace old {
double computeLoglikelihoodNaiveUtree(AnnotatedNetwork &ann_network, int incremental,
        int update_pmatrices, std::vector<double> *treewise_logl = nullptr);
double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices,
        bool update_reticulation_probs = false, std::vector<double> *treewise_logl = nullptr);
}
}
