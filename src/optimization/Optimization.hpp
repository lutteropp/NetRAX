#pragma once

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax
{
    
void optimizeBranches(AnnotatedNetwork &ann_network, bool silent = true, bool restricted_total_iters = false);
void optimizeModel(AnnotatedNetwork &ann_network, bool silent = true);
void optimizeReticulationProbs(AnnotatedNetwork &ann_network, bool silent = true);
void optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt = false, bool silent = true);

} // namespace netrax


