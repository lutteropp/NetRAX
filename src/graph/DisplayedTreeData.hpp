#pragma once

#include <iostream>
#include <limits>
#include <vector>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "RangeInfo.hpp"
#include "ReticulationConfigSet.hpp"
#include "TreeLoglData.hpp"

namespace netrax {

struct DisplayedTreeData {
  TreeLoglData treeLoglData;
  std::vector<double*> clv_vector;
  std::vector<unsigned int*> scale_buffer;
  const std::vector<ClvRangeInfo>& clvInfo;
  const std::vector<ScaleBufferRangeInfo>& scaleBufferInfo;
  bool isTip = false;
  bool clv_valid = false;

  DisplayedTreeData(
      pllmod_treeinfo_t* treeinfo,
      const std::vector<ClvRangeInfo>& clvRangeInfo,
      const std::vector<ScaleBufferRangeInfo>& scaleBufferRangeInfo,
      size_t max_reticulations);
  DisplayedTreeData(
      pllmod_treeinfo_t* treeinfo,
      const std::vector<ClvRangeInfo>& clvRangeInfo,
      const std::vector<ScaleBufferRangeInfo>& scaleBufferRangeInfo,
      std::vector<double*> tip_clv_vector, size_t max_reticulations);
  DisplayedTreeData(DisplayedTreeData&& rhs);
  DisplayedTreeData(const DisplayedTreeData& rhs);
  DisplayedTreeData& operator=(DisplayedTreeData&& rhs);
  DisplayedTreeData& operator=(const DisplayedTreeData& rhs);
  ~DisplayedTreeData();
};

}  // namespace netrax
