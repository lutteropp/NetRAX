#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "ScoreImprovement.hpp"
#include "../moves/MoveType.hpp"

namespace netrax {

void wavesearch(AnnotatedNetwork& ann_network, BestNetworkData* bestNetworkData, const std::vector<MoveType>& typesBySpeed, const std::vector<MoveType>& typesBySpeedGoodStart, bool silent = true);

}