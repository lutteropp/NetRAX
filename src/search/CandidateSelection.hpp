#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../optimization/NetworkState.hpp"
#include "ScoreImprovement.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/PseudoLoglikelihood.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../moves/MoveType.hpp"
#include "../moves/Moves.hpp"
#include "../io/NetworkIO.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/Optimization.hpp"

namespace netrax {

double applyBestCandidate(AnnotatedNetwork& ann_network, MoveType type, const std::vector<MoveType>& typesBySpeed, double* best_score, BestNetworkData* bestNetworkData, bool enforce, bool silent);

}