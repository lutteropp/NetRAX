#pragma once

#include <random>
#include <raxml-ng/main.hpp>
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
    
void run_single_start_waves(const NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng);
void run_random(const NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng);

void optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt = false, bool silent = true);

}