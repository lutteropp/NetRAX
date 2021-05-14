#pragma once

#include "../graph/AnnotatedNetwork.hpp"

#include <unordered_set>

namespace netrax
{
    
void optimizeBranches(AnnotatedNetwork &ann_network, bool silent = true, bool restricted_total_iters = false);
void optimizeBranchesCandidates(AnnotatedNetwork &ann_network, std::unordered_set<size_t> brlenopt_candidates, bool silent = true, bool restricted_total_iters = false);
void optimizeModel(AnnotatedNetwork &ann_network, bool silent = true);
void optimizeReticulationProbs(AnnotatedNetwork &ann_network, bool silent = true);
void optimizeAllNonTopology(AnnotatedNetwork &ann_network, bool extremeOpt = false, bool silent = true);

} // namespace netrax


