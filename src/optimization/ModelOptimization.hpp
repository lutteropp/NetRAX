#pragma once

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
    double optimize_params(AnnotatedNetwork& ann_network, double lh_epsilon);
}