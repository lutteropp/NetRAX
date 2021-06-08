#pragma once

#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

void judgeNetwork(AnnotatedNetwork &inferredNetwork,
                  AnnotatedNetwork &trueNetwork);

void run_single_start_waves(NetraxOptions &netraxOptions,
                            const RaxmlInstance &instance,
                            const std::vector<MoveType> &typesBySpeed,
                            std::mt19937 &rng, bool silent,
                            bool print_progress);
void run_random(NetraxOptions &netraxOptions, const RaxmlInstance &instance,
                const std::vector<MoveType> &typesBySpeed, std::mt19937 &rng,
                bool silent, bool print_progress);

}  // namespace netrax