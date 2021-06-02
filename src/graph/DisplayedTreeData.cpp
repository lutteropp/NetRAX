#include "DisplayedTreeData.hpp"

#include <iostream>
#include <stdexcept>

namespace netrax {

DisplayedTreeData::DisplayedTreeData(
    pllmod_treeinfo_t *treeinfo, const std::vector<ClvRangeInfo> &clvRangeInfo,
    const std::vector<ScaleBufferRangeInfo> &scaleBufferRangeInfo,
    size_t max_reticulations)
    : treeLoglData(treeinfo->partition_count, max_reticulations),
      clvInfo(clvRangeInfo),
      scaleBufferInfo(scaleBufferRangeInfo) {  // inner node
  clv_vector = std::vector<double *>(treeinfo->partition_count, nullptr);
  for (size_t p = 0; p < treeinfo->partition_count; ++p) {
    // skip remote partitions
    if (!treeinfo->partitions[p]) {
      continue;
    }
    clv_vector[p] = create_single_empty_clv(clvRangeInfo[p]);
  }
  scale_buffer =
      std::vector<unsigned int *>(treeinfo->partition_count, nullptr);
  for (size_t p = 0; p < treeinfo->partition_count; ++p) {
    // skip remote partitions
    if (!treeinfo->partitions[p]) {
      continue;
    }
    scale_buffer[p] = create_single_empty_scale_buffer(scaleBufferRangeInfo[p]);
  }
}

DisplayedTreeData::DisplayedTreeData(
    pllmod_treeinfo_t *treeinfo, const std::vector<ClvRangeInfo> &clvRangeInfo,
    const std::vector<ScaleBufferRangeInfo> &scaleBufferRangeInfo,
    std::vector<double *> tip_clv_vector, size_t max_reticulations)
    : treeLoglData(treeinfo->partition_count, max_reticulations),
      clvInfo(clvRangeInfo),
      scaleBufferInfo(scaleBufferRangeInfo) {  // tip node
  clv_vector = std::vector<double *>(treeinfo->partition_count, nullptr);
  for (size_t p = 0; p < treeinfo->partition_count; ++p) {
    // skip remote partitions
    if (!treeinfo->partitions[p]) {
      continue;
    }
    clv_vector[p] = tip_clv_vector[p];
  }
  isTip = true;
  scale_buffer =
      std::vector<unsigned int *>(treeinfo->partition_count, nullptr);
}

DisplayedTreeData::DisplayedTreeData(DisplayedTreeData &&rhs)
    : treeLoglData{rhs.treeLoglData},
      clv_vector{rhs.clv_vector},
      scale_buffer{rhs.scale_buffer},
      clvInfo{rhs.clvInfo},
      scaleBufferInfo{rhs.scaleBufferInfo},
      isTip{rhs.isTip},
      clv_valid{rhs.clv_valid} {
  rhs.clv_vector.clear();
  rhs.scale_buffer.clear();
}

DisplayedTreeData::DisplayedTreeData(const DisplayedTreeData &rhs)
    : treeLoglData{rhs.treeLoglData},
      clvInfo{rhs.clvInfo},
      scaleBufferInfo{rhs.scaleBufferInfo},
      isTip{rhs.isTip},
      clv_valid{rhs.clv_valid} {
  if (isTip) {
    clv_vector = rhs.clv_vector;
    scale_buffer = rhs.scale_buffer;
  } else {
    clv_vector = std::vector<double *>(rhs.clv_vector.size(), nullptr);
    assert(rhs.clvInfo.size() == rhs.clvInfo.size());
    for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
      clv_vector[p] =
          clone_single_clv_vector(rhs.clvInfo[p], rhs.clv_vector[p]);
    }
    scale_buffer =
        std::vector<unsigned int *>(rhs.scale_buffer.size(), nullptr);
    assert(rhs.scaleBufferInfo.size() == rhs.scale_buffer.size());
    for (size_t p = 0; p < rhs.scale_buffer.size(); ++p) {
      scale_buffer[p] = clone_single_scale_buffer(rhs.scaleBufferInfo[p],
                                                  rhs.scale_buffer[p]);
    }
  }
}

DisplayedTreeData &DisplayedTreeData::operator=(DisplayedTreeData &&rhs) {
  if (this != &rhs) {
    if (!isTip) {
      for (size_t p = 0; p < clv_vector.size(); ++p) {
        pll_aligned_free(clv_vector[p]);
      }
    }
    for (size_t p = 0; p < scale_buffer.size(); ++p) {
      free(scale_buffer[p]);
    }
    treeLoglData = rhs.treeLoglData;
    clv_vector = rhs.clv_vector;
    scale_buffer = rhs.scale_buffer;

    rhs.clv_vector.clear();
    rhs.scale_buffer.clear();
    isTip = rhs.isTip;
    clv_valid = rhs.clv_valid;
  }
  return *this;
}

DisplayedTreeData &DisplayedTreeData::operator=(const DisplayedTreeData &rhs) {
  if (this != &rhs) {
    treeLoglData = rhs.treeLoglData;
    if ((clv_vector.size() == rhs.clv_vector.size()) &&
        (clvInfo == rhs.clvInfo) && (isTip == rhs.isTip)) {  // simply overwrite
      if (rhs.isTip) {
        clv_vector = rhs.clv_vector;
      } else {
        assert(rhs.clvInfo.size() == rhs.clvInfo.size());
        for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
          if (!rhs.clv_vector[p]) {
            continue;
          }
          assert(clv_vector[p]);
          memcpy(clv_vector[p], rhs.clv_vector[p],
                 rhs.clvInfo[p].inner_clv_num_entries * sizeof(double));
        }
      }
    } else {
      if (isTip) {
        clv_vector = rhs.clv_vector;
      } else {
        for (size_t p = 0; p < clv_vector.size(); ++p) {
          pll_aligned_free(clv_vector[p]);
        }
        if (rhs.isTip) {
          clv_vector = rhs.clv_vector;
        } else {
          assert(rhs.clvInfo.size() == rhs.clvInfo.size());
          clv_vector = std::vector<double *>(rhs.clv_vector.size(), nullptr);
          for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
            clv_vector[p] =
                clone_single_clv_vector(rhs.clvInfo[p], rhs.clv_vector[p]);
          }
        }
      }
    }
    if ((scale_buffer.size() == rhs.scale_buffer.size()) &&
        (scaleBufferInfo == rhs.scaleBufferInfo)) {  // simply overwrite
      assert(rhs.scaleBufferInfo.size() == rhs.scale_buffer.size());
      for (size_t p = 0; p < rhs.scale_buffer.size(); ++p) {
        if (!rhs.scale_buffer[p]) {
          continue;
        }
        assert(scale_buffer[p]);
        memcpy(scale_buffer[p], rhs.scale_buffer[p],
               rhs.scaleBufferInfo[p].scaler_size * sizeof(unsigned int));
      }
    } else {
      for (size_t p = 0; p < scale_buffer.size(); ++p) {
        free(scale_buffer[p]);
      }
      scale_buffer =
          std::vector<unsigned int *>(rhs.scale_buffer.size(), nullptr);
      assert(rhs.scaleBufferInfo.size() == rhs.scale_buffer.size());
      for (size_t p = 0; p < rhs.scale_buffer.size(); ++p) {
        scale_buffer[p] = clone_single_scale_buffer(rhs.scaleBufferInfo[p],
                                                    rhs.scale_buffer[p]);
      }
    }
    isTip = rhs.isTip;
    clv_valid = rhs.clv_valid;
  }
  return *this;
}

DisplayedTreeData::~DisplayedTreeData() {
  if (!isTip) {
    for (size_t p = 0; p < clv_vector.size(); ++p) {
      pll_aligned_free(clv_vector[p]);
    }
    for (size_t p = 0; p < scale_buffer.size(); ++p) {
      free(scale_buffer[p]);
    }
  }
}

}  // namespace netrax