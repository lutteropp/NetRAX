#pragma once

#include <vector>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

struct DisplayedTreeData {
    size_t tree_idx = 0;
    double tree_logl = 0.0;
    double tree_logprob = 0.0;
    double** tree_clv_vectors = nullptr;
    unsigned int** tree_scale_buffers = nullptr;
    std::vector<double> tree_persite_logl;
};

struct ClvRangeInfo {
    unsigned int start = 0;
    unsigned int end = 0;
    size_t inner_clv_num_entries = 0;
    bool operator==(const ClvRangeInfo& other) const {
        return ((start == other.start) && (end == other.end) && (inner_clv_num_entries == other.inner_clv_num_entries));
    }
};

struct ScaleBufferRangeInfo {
    unsigned int scaler_size = 0;
    unsigned int num_scale_buffers = 0;
    bool operator==(const ScaleBufferRangeInfo& other) const {
        return ((scaler_size == other.scaler_size) && (num_scale_buffers == other.num_scale_buffers));
    }
};

ClvRangeInfo get_clv_range(pll_partition_t* partition);
bool clv_entries_equal(ClvRangeInfo rangeInfo, double** clv1, double** clv2);
double** clone_clv_vector(pll_partition_t* partition, double** clv);
void delete_cloned_clv_vector(ClvRangeInfo rangeInfo, double** clv);
void delete_cloned_clv_vector(pll_partition_t* partition, double** clv);
void assign_clv_entries(pll_partition_t* partition, double** from_clv, double** to_clv);

ScaleBufferRangeInfo get_scale_buffer_range(pll_partition_t* partition);
bool scale_buffer_entries_equal(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer_1, unsigned int** scale_buffer_2);
unsigned int** clone_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer);
void delete_cloned_scale_buffer(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer);
void delete_cloned_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer);
void assign_scale_buffer_entries(pll_partition_t* partition, unsigned int** from_scale_buffer, unsigned int** to_scale_buffer);
}