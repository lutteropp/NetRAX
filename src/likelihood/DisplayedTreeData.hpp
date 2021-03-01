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
    unsigned int alignment = 0;
    unsigned int total_num_clvs = 0;
    unsigned int start = 0;
    unsigned int end = 0;
    size_t inner_clv_num_entries = 0;
    bool operator==(const ClvRangeInfo& other) const {
        return ((alignment == other.alignment) && (total_num_clvs == other.total_num_clvs) && (start == other.start) && (end == other.end) && (inner_clv_num_entries == other.inner_clv_num_entries));
    }
};

struct ScaleBufferRangeInfo {
    unsigned int scaler_size = 0;
    unsigned int num_scale_buffers = 0;
    bool operator==(const ScaleBufferRangeInfo& other) const {
        return ((scaler_size == other.scaler_size) && (num_scale_buffers == other.num_scale_buffers));
    }
};

void print_clv(ClvRangeInfo rangeInfo, double ** clv);
ClvRangeInfo get_clv_range(pll_partition_t* partition);
bool clv_entries_equal(ClvRangeInfo rangeInfo, double** clv1, double** clv2);
double* create_single_empty_clv(ClvRangeInfo rangeInfo);
double** create_empty_clv_vector(ClvRangeInfo rangeInfo);
double** clone_clv_vector(pll_partition_t* partition, double** clv);
void delete_cloned_clv_vector(ClvRangeInfo rangeInfo, double** clv);
void delete_cloned_clv_vector(pll_partition_t* partition, double** clv);
void assign_clv_entries(pll_partition_t* partition, double** from_clv, double** to_clv);

ScaleBufferRangeInfo get_scale_buffer_range(pll_partition_t* partition);
bool scale_buffer_entries_equal(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer_1, unsigned int** scale_buffer_2);
unsigned int * create_single_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo);
unsigned int ** create_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo);
unsigned int** clone_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer);
void delete_cloned_scale_buffer(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer);
void delete_cloned_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer);
void assign_scale_buffer_entries(pll_partition_t* partition, unsigned int** from_scale_buffer, unsigned int** to_scale_buffer);

enum class ReticulationState {
    DONT_CARE = 0,
    TAKE_FIRST_PARENT = 1,
    TAKE_SECOND_PARENT = 2
};

struct DisplayedTreeClvData {
    bool is_tip = false;
    double* clv_vector = nullptr;
    unsigned int* scale_buffer = nullptr;
    std::vector<ReticulationState> reticulationChoices;

    DisplayedTreeClvData(ClvRangeInfo clvRangeInfo, ScaleBufferRangeInfo scaleBufferRangeInfo, size_t max_reticulations) { // inner node
        is_tip = false;
        reticulationChoices.resize(max_reticulations);
        clv_vector = create_single_empty_clv(clvRangeInfo);
        scale_buffer = create_single_empty_scale_buffer(scaleBufferRangeInfo);
    }

    DisplayedTreeClvData(double* tip_clv_vector, size_t max_reticulations) { // tip node
        is_tip = true;
        reticulationChoices.resize(max_reticulations);
        clv_vector = tip_clv_vector;
        scale_buffer = nullptr;
    }

    ~DisplayedTreeClvData() {
        if (!is_tip) {
            pll_aligned_free(clv_vector);
            free(scale_buffer);
        }
    }
};

double computeReticulationChoicesLogProb(const std::vector<ReticulationState>& choices, const std::vector<double>& reticulationProbs);
bool reticulationChoicesCompatible(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
std::vector<ReticulationState> combineReticulationChoices(std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);


}