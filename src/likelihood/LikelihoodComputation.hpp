/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "../graph/Common.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../RaxmlWrapper.hpp"
#include <raxml-ng/TreeInfo.hpp>

namespace netrax {

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental = 1, int update_pmatrices = 1);

double computeLoglikelihoodNaiveUtree(AnnotatedNetwork &ann_network, int incremental,
        int update_pmatrices, std::vector<double> *treewise_logl = nullptr);

void setup_pmatrices(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices);
double displayed_tree_logprob(AnnotatedNetwork &ann_network, size_t tree_index);

size_t get_param_count(AnnotatedNetwork& ann_network);
size_t get_sample_size(AnnotatedNetwork& ann_network);
double aic(AnnotatedNetwork &ann_network, double logl);
double aicc(AnnotatedNetwork &ann_network, double logl);
double bic(AnnotatedNetwork &ann_network, double logl);
double scoreNetwork(AnnotatedNetwork &ann_network);

}
