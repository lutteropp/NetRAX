#pragma once

#include <vector>
#include <limits>

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
bool clv_single_entries_equal(ClvRangeInfo rangeInfo, double* clv1, double* clv2);
bool clv_entries_equal(ClvRangeInfo rangeInfo, double** clv1, double** clv2);
double* create_single_empty_clv(ClvRangeInfo rangeInfo);
double** create_empty_clv_vector(ClvRangeInfo rangeInfo);
double* clone_single_clv_vector(ClvRangeInfo clvInfo, double* clv);
double** clone_clv_vector(pll_partition_t* partition, double** clv);
void delete_cloned_clv_vector(ClvRangeInfo rangeInfo, double** clv);
void delete_cloned_clv_vector(pll_partition_t* partition, double** clv);
void assign_clv_entries(pll_partition_t* partition, double** from_clv, double** to_clv);

ScaleBufferRangeInfo get_scale_buffer_range(pll_partition_t* partition);
bool scale_buffer_single_entries_equal(ScaleBufferRangeInfo rangeInfo, unsigned int* scale_buffer_1, unsigned int* scale_buffer_2);
bool scale_buffer_entries_equal(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer_1, unsigned int** scale_buffer_2);
unsigned int * create_single_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo);
unsigned int ** create_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo);
unsigned int* clone_single_scale_buffer(ScaleBufferRangeInfo scaleBufferInfo, unsigned int* scale_buffer);
unsigned int** clone_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer);
void delete_cloned_scale_buffer(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer);
void delete_cloned_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer);
void assign_scale_buffer_entries(pll_partition_t* partition, unsigned int** from_scale_buffer, unsigned int** to_scale_buffer);

enum class ReticulationState {
    DONT_CARE = 0,
    TAKE_FIRST_PARENT = 1,
    TAKE_SECOND_PARENT = 2
};

struct ReticulationConfigSet {
    std::vector<std::vector<ReticulationState> > configs;
    size_t max_reticulations = 0;

    ReticulationConfigSet(size_t max_reticulations) : max_reticulations(max_reticulations) {
    }
};

struct DisplayedTreeData {
    bool tree_logl_valid = false;
    bool tree_logprob_valid = false;
    double* clv_vector = nullptr;
    unsigned int* scale_buffer = nullptr;
    ClvRangeInfo clvInfo;
    ScaleBufferRangeInfo scaleBufferInfo;

    double tree_logl = -std::numeric_limits<double>::infinity();
    double tree_logprob = 0;
    ReticulationConfigSet reticulationChoices;

    DisplayedTreeData(ClvRangeInfo clvRangeInfo, ScaleBufferRangeInfo scaleBufferRangeInfo, size_t max_reticulations) : reticulationChoices(max_reticulations) { // inner node
        clv_vector = create_single_empty_clv(clvRangeInfo);
        scale_buffer = create_single_empty_scale_buffer(scaleBufferRangeInfo);
        this->clvInfo = clvRangeInfo;
        this->scaleBufferInfo = scaleBufferRangeInfo;
    }

    DisplayedTreeData(double* tip_clv_vector, size_t max_reticulations) : reticulationChoices(max_reticulations) { // tip node
        clv_vector = tip_clv_vector;
        scale_buffer = nullptr;
    }

    DisplayedTreeData(DisplayedTreeData&& rhs)
      : tree_logl_valid{rhs.tree_logl_valid}, tree_logprob_valid{rhs.tree_logprob_valid}, clv_vector{rhs.clv_vector}, scale_buffer{rhs.scale_buffer}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}, tree_logl{rhs.tree_logl}, tree_logprob{rhs.tree_logprob}, reticulationChoices{rhs.reticulationChoices}
    {
        rhs.clv_vector = nullptr;
        rhs.scale_buffer = nullptr;
    }

    DisplayedTreeData(const DisplayedTreeData& rhs)
      : tree_logl_valid{rhs.tree_logl_valid}, tree_logprob_valid{rhs.tree_logprob_valid}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}, tree_logl{rhs.tree_logl}, tree_logprob{rhs.tree_logprob}, reticulationChoices{rhs.reticulationChoices}
    {
        clv_vector = clone_single_clv_vector(rhs.clvInfo, rhs.clv_vector);
        scale_buffer = clone_single_scale_buffer(rhs.scaleBufferInfo, rhs.scale_buffer);
    }

    DisplayedTreeData& operator =(DisplayedTreeData&& rhs)
    {
        if (this != &rhs)
        {
            pll_aligned_free(clv_vector);
            free(scale_buffer);
            tree_logl_valid = rhs.tree_logl_valid;
            tree_logprob_valid = rhs.tree_logprob_valid;
            clv_vector = rhs.clv_vector;
            scale_buffer = rhs.scale_buffer;
            clvInfo = rhs.clvInfo;
            scaleBufferInfo = rhs.scaleBufferInfo;
            tree_logl = rhs.tree_logl;
            tree_logprob = rhs.tree_logprob;
            reticulationChoices = std::move(rhs.reticulationChoices);

            rhs.clv_vector = nullptr;
            rhs.scale_buffer = nullptr;
        }
        return *this;
    }

    DisplayedTreeData& operator =(const DisplayedTreeData& rhs)
    {
        if (this != &rhs)
        {
            tree_logl_valid = rhs.tree_logl_valid;
            tree_logprob_valid = rhs.tree_logprob_valid;
            tree_logl = rhs.tree_logl;
            tree_logprob = rhs.tree_logprob;
            reticulationChoices = rhs.reticulationChoices;
            if ((clv_vector && rhs.clv_vector) && (clvInfo == rhs.clvInfo)) { // simply overwrite
                memcpy(clv_vector, rhs.clv_vector, clvInfo.inner_clv_num_entries * sizeof(double));
            } else {
                pll_aligned_free(clv_vector);
                clv_vector = clone_single_clv_vector(rhs.clvInfo, rhs.clv_vector);
            }
            if ((scale_buffer && rhs.scale_buffer) && (scaleBufferInfo == rhs.scaleBufferInfo)) { // simply overwrite
                memcpy(scale_buffer, rhs.scale_buffer, scaleBufferInfo.scaler_size * sizeof(unsigned int));
            } else {
                free(scale_buffer);
                scale_buffer = clone_single_scale_buffer(rhs.scaleBufferInfo, rhs.scale_buffer);
            }
            clvInfo = rhs.clvInfo;
            scaleBufferInfo = rhs.scaleBufferInfo;
        }
        return *this;
    }

    ~DisplayedTreeData() {
        pll_aligned_free(clv_vector);
        free(scale_buffer);
    }
};

//double computeReticulationChoicesLogProb(const std::vector<ReticulationState>& choices, const std::vector<double>& reticulationProbs);
double computeReticulationConfigLogProb(const ReticulationConfigSet& choices, const std::vector<double>& reticulationProbs);

//bool reticulationChoicesCompatible(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
bool reticulationConfigsCompatible(const ReticulationConfigSet& left, const ReticulationConfigSet& right);

//void printReticulationChoices(const std::vector<ReticulationState>& reticulationChoices);
void printReticulationChoices(const ReticulationConfigSet& reticulationChoices);

//std::vector<ReticulationState> combineReticulationChoices(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
ReticulationConfigSet combineReticulationChoices(const ReticulationConfigSet& left, const ReticulationConfigSet& right);

void simplifyReticulationChoices(ReticulationConfigSet& res);

}