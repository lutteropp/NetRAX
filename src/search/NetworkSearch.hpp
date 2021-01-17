#pragma once

#include <random>
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
    
void run_single_start_waves(NetraxOptions& netraxOptions, std::mt19937& rng);
void run_random(NetraxOptions& netraxOptions, std::mt19937& rng);

void optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt = false);

}