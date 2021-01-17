
#pragma once

#include "../graph/Network.hpp"
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

#include <unordered_set>

extern "C" {
#include <libpll/pll_tree.h>
}

namespace netrax {

void optimizeReticulationProbs(AnnotatedNetwork &ann_network);

}
