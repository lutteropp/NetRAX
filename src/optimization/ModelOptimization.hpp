#pragma once

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
    void optimizeModel(AnnotatedNetwork &ann_network, bool silent = true);
}