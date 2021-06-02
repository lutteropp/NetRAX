#pragma once

#include <cstddef>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

struct ClvRangeInfo {
  unsigned int alignment = 0;
  unsigned int total_num_clvs = 0;
  unsigned int start = 0;
  unsigned int end = 0;
  size_t inner_clv_num_entries = 0;
  bool operator==(const ClvRangeInfo &other) const {
    return ((alignment == other.alignment) &&
            (total_num_clvs == other.total_num_clvs) &&
            (start == other.start) && (end == other.end) &&
            (inner_clv_num_entries == other.inner_clv_num_entries));
  }
};

struct ScaleBufferRangeInfo {
  unsigned int scaler_size = 0;
  unsigned int num_scale_buffers = 0;
  bool operator==(const ScaleBufferRangeInfo &other) const {
    return ((scaler_size == other.scaler_size) &&
            (num_scale_buffers == other.num_scale_buffers));
  }
};

bool single_clv_is_all_zeros(ClvRangeInfo rangeInfo, double *clv);

void print_node_clv(ClvRangeInfo rangeInfo, double *clv);
void print_clv(ClvRangeInfo rangeInfo, double **clv);
ClvRangeInfo get_clv_range(pll_partition_t *partition);
bool clv_single_entries_equal(ClvRangeInfo rangeInfo, double *clv1,
                              double *clv2);
bool clv_entries_equal(ClvRangeInfo rangeInfo, double **clv1, double **clv2);
double *create_single_empty_clv(ClvRangeInfo rangeInfo);
double **create_empty_clv_vector(ClvRangeInfo rangeInfo);
double *clone_single_clv_vector(ClvRangeInfo clvInfo, double *clv);
double **clone_clv_vector(pll_partition_t *partition, double **clv);
void delete_cloned_clv_vector(ClvRangeInfo rangeInfo, double **clv);
void delete_cloned_clv_vector(pll_partition_t *partition, double **clv);
void assign_clv_entries(pll_partition_t *partition, double **from_clv,
                        double **to_clv);

void print_node_scaler(ScaleBufferRangeInfo rangeInfo,
                       unsigned int *scale_buffer);
ScaleBufferRangeInfo get_scale_buffer_range(pll_partition_t *partition);
bool scale_buffer_single_entries_equal(ScaleBufferRangeInfo rangeInfo,
                                       unsigned int *scale_buffer_1,
                                       unsigned int *scale_buffer_2);
bool scale_buffer_entries_equal(ScaleBufferRangeInfo rangeInfo,
                                unsigned int **scale_buffer_1,
                                unsigned int **scale_buffer_2);
unsigned int *create_single_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo);
unsigned int **create_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo);
unsigned int *clone_single_scale_buffer(ScaleBufferRangeInfo scaleBufferInfo,
                                        unsigned int *scale_buffer);
unsigned int **clone_scale_buffer(pll_partition_t *partition,
                                  unsigned int **scale_buffer);
void delete_cloned_scale_buffer(ScaleBufferRangeInfo rangeInfo,
                                unsigned int **scale_buffer);
void delete_cloned_scale_buffer(pll_partition_t *partition,
                                unsigned int **scale_buffer);
void assign_scale_buffer_entries(pll_partition_t *partition,
                                 unsigned int **from_scale_buffer,
                                 unsigned int **to_scale_buffer);

}  // namespace netrax