#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../optimization/MoveType.hpp"

namespace netrax {

void scrambleNetwork(AnnotatedNetwork& ann_network, MoveType type, size_t scramble_cnt);

}