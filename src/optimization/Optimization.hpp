#pragma once

#include "../graph/AnnotatedNetwork.hpp"

#include <unordered_set>

namespace netrax
{
    
void optimizeBranches(AnnotatedNetwork &ann_network, double brlen_smooth_factor = 100, bool silent = true, bool restricted_total_iters = false);
void optimizeBranchesCandidates(AnnotatedNetwork &ann_network, std::unordered_set<size_t> brlenopt_candidates, double brlen_smooth_factor = 0.25, bool silent = true, bool restricted_total_iters = false);
void optimizeModel(AnnotatedNetwork &ann_network, bool silent = true);
void optimizeReticulationProbs(AnnotatedNetwork &ann_network, bool silent = true);

enum class OptimizeAllNonTopologyType {
    QUICK = 0,
    NORMAL = 1,
    SLOW = 2
};

void optimizeAllNonTopology(AnnotatedNetwork &ann_network, OptimizeAllNonTopologyType type, bool silent = true);

} // namespace netrax


