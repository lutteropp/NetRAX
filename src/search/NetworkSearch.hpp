#pragma once

#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {
    
void run_single_start_waves(NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng);
void run_random(NetraxOptions& netraxOptions, const RaxmlInstance& instance, const std::vector<MoveType>& typesBySpeed, std::mt19937& rng);

}