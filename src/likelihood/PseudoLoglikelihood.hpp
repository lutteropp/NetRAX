#pragma once

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
    double computePseudoLoglikelihood(AnnotatedNetwork& ann_network, int incremental = 1, int update_pmatrices = 1);
}