#pragma once

#include "graph/AnnotatedNetwork.hpp"
#include "NetraxOptions.hpp"
#include "io/NetworkIO.hpp"

#include <unordered_map>
#include <string>

namespace netrax
{
    enum class NetworkDistanceType
    {
        UNROOTED_SOFTWIRED_DISTANCE = 0,
        ROOTED_SOFTWIRED_DISTANCE = 1,
        UNROOTED_HARDWIRED_DISTANCE = 2,
        ROOTED_HARDWIRED_DISTANCE = 3,

        ROOTED_DISPLAYED_TREES_DISTANCE = 4,
        UNROOTED_DISPLAYED_TREES_DISTANCE = 5
    };

    double get_network_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork &ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int, NetworkDistanceType type);
} // namespace netrax