#pragma once

#include "../graph/AnnotatedNetwork.hpp"
#include "../moves/MoveType.hpp"

namespace netrax {

void scrambleNetwork(AnnotatedNetwork& ann_network, MoveType type, size_t scramble_cnt);

}