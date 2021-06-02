#pragma once

#include <vector>

namespace netrax {

struct DisplayedTreeData;
struct ClvRangeInfo;
struct ScaleBufferRangeInfo;

struct NodeDisplayedTreeData {
  std::vector<DisplayedTreeData> displayed_trees;
  size_t num_active_displayed_trees = 0;
  size_t partition_count = 0;
  std::vector<ClvRangeInfo> clvInfo;
  std::vector<ScaleBufferRangeInfo> scaleBufferInfo;

  bool operator==(const NodeDisplayedTreeData& rhs) const;
  bool operator!=(const NodeDisplayedTreeData& rhs) const;

  NodeDisplayedTreeData(NodeDisplayedTreeData&& rhs);
  NodeDisplayedTreeData(const NodeDisplayedTreeData& rhs);
  NodeDisplayedTreeData() = default;

  NodeDisplayedTreeData& operator=(NodeDisplayedTreeData&& rhs);
  NodeDisplayedTreeData& operator=(const NodeDisplayedTreeData& rhs);
};

}  // namespace netrax