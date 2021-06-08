#pragma once

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

double computeLoglikelihoodNaiveUtree(
    AnnotatedNetwork &ann_network, int incremental, int update_pmatrices,
    std::vector<double> *treewise_logl = nullptr);

}