#pragma once

#include "NetraxOptions.hpp"
#include "graph/AnnotatedNetwork.hpp"
#include "io/NetworkIO.hpp"

#include <string>
#include <unordered_map>

namespace netrax {
enum class NetworkDistanceType {
  UNROOTED_SOFTWIRED_DISTANCE = 0,
  ROOTED_SOFTWIRED_DISTANCE = 1,
  UNROOTED_HARDWIRED_DISTANCE = 2,
  ROOTED_HARDWIRED_DISTANCE = 3,

  ROOTED_DISPLAYED_TREES_DISTANCE = 4,
  UNROOTED_DISPLAYED_TREES_DISTANCE = 5,

  ROOTED_TRIPARTITION_DISTANCE = 6,

  ROOTED_PATH_MULTIPLICITY_DISTANCE = 7,

  ROOTED_NESTED_LABELS_DISTANCE = 8
};

double get_network_distance(
    AnnotatedNetwork &ann_network_1, AnnotatedNetwork &ann_network_2,
    std::unordered_map<std::string, unsigned int> &label_to_int,
    NetworkDistanceType type);
}  // namespace netrax