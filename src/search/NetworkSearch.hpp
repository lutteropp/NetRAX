#pragma once

#include <random>
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
    
void run_single_start_waves(NetraxOptions& netraxOptions, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng);
void run_random(NetraxOptions& netraxOptions, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng);

void optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt = false, bool silent = true);

}