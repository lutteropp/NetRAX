#include "NodeDisplayedTreeData.hpp"
#include "DisplayedTreeData.hpp"

namespace netrax {

bool NodeDisplayedTreeData::operator==(const NodeDisplayedTreeData &rhs) const {
  if (num_active_displayed_trees != rhs.num_active_displayed_trees) {
    return false;
  }
  for (size_t i = 0; i < num_active_displayed_trees; ++i) {
    for (size_t j = 0; j < partition_count; ++j) {
      if (!clv_single_entries_equal(clvInfo[j],
                                    displayed_trees[i].clv_vector[j],
                                    rhs.displayed_trees[i].clv_vector[j])) {
        return false;
      }
      if (!scale_buffer_single_entries_equal(
              scaleBufferInfo[j], displayed_trees[i].scale_buffer[j],
              rhs.displayed_trees[i].scale_buffer[j])) {
        return false;
      }
    }
  }
  return true;
}

bool NodeDisplayedTreeData::operator!=(const NodeDisplayedTreeData &rhs) const {
  return !operator==(rhs);
}

NodeDisplayedTreeData::NodeDisplayedTreeData(NodeDisplayedTreeData &&rhs)
    : displayed_trees{rhs.displayed_trees},
      num_active_displayed_trees{rhs.num_active_displayed_trees},
      partition_count{rhs.partition_count},
      clvInfo{rhs.clvInfo},
      scaleBufferInfo{rhs.scaleBufferInfo} {
  rhs.num_active_displayed_trees = 0;
  rhs.displayed_trees = std::vector<DisplayedTreeData>();
}

NodeDisplayedTreeData::NodeDisplayedTreeData(const NodeDisplayedTreeData &rhs)
    : num_active_displayed_trees{rhs.num_active_displayed_trees},
      partition_count{rhs.partition_count},
      clvInfo{rhs.clvInfo},
      scaleBufferInfo{rhs.scaleBufferInfo} {
  displayed_trees = rhs.displayed_trees;
}

NodeDisplayedTreeData &NodeDisplayedTreeData::operator=(
    NodeDisplayedTreeData &&rhs) {
  if (this != &rhs) {
    displayed_trees = std::move(rhs.displayed_trees);
    num_active_displayed_trees = rhs.num_active_displayed_trees;
    partition_count = rhs.partition_count;
    rhs.num_active_displayed_trees = 0;
    clvInfo = rhs.clvInfo;
    scaleBufferInfo = rhs.scaleBufferInfo;
  }
  return *this;
}

NodeDisplayedTreeData &NodeDisplayedTreeData::operator=(
    const NodeDisplayedTreeData &rhs) {
  if (this != &rhs) {
    for (size_t i = 0;
         i < std::min(displayed_trees.size(), rhs.displayed_trees.size());
         ++i) {
      displayed_trees[i] = rhs.displayed_trees[i];
    }
    for (size_t i =
             std::min(displayed_trees.size(), rhs.displayed_trees.size());
         i < rhs.displayed_trees.size(); ++i) {
      displayed_trees.emplace_back(rhs.displayed_trees[i]);
    }
    num_active_displayed_trees = rhs.num_active_displayed_trees;
    partition_count = rhs.partition_count;
    clvInfo = rhs.clvInfo;
    scaleBufferInfo = rhs.scaleBufferInfo;
  }
  return *this;
}

}  // namespace netrax