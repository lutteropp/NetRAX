#pragma once

#include "graph/AnnotatedNetwork.hpp"
#include "NetraxOptions.hpp"
#include "io/NetworkIO.hpp"

namespace netrax {
    enum class NetworkDistanceType {
        UNROOTED_SOFTWIRED_DISTANCE = 0,
        ROOTED_SOFTWIRED_DISTANCE = 1,
        UNROOTED_HARDWIRED_DISTANCE = 2,
        ROOTED_HARDWIRED_DISTANCE = 3
    };

    double get_network_distance(AnnotatedNetwork& ann_network_1, AnnotatedNetwork& ann_network_2, NetworkDistanceType type);
}