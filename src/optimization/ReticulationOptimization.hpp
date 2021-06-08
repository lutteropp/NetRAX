
#pragma once

#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/Network.hpp"

#include <unordered_set>

extern "C" {
#include <libpll/pll_tree.h>
}

namespace netrax {

double optimize_reticulations(AnnotatedNetwork &ann_network, int max_iters);
double optimize_reticulation(AnnotatedNetwork &ann_network,
                             size_t reticulation_index);

}  // namespace netrax
