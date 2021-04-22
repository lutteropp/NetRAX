#pragma once

#include <vector>
#include <limits>
#include <iostream>

#include <raxml-ng/main.hpp>

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

bool single_clv_is_all_zeros(ClvRangeInfo rangeInfo, double* clv);


void print_node_clv(ClvRangeInfo rangeInfo, double * clv);
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

void print_node_scaler(ScaleBufferRangeInfo rangeInfo, unsigned int * scale_buffer);
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
    TAKE_SECOND_PARENT = 2,
    INVALID = 3
};

struct ReticulationConfigSet {
    std::vector<std::vector<ReticulationState> > configs;
    size_t max_reticulations = 0;

    bool operator==(const ReticulationConfigSet& other) const {
        if (max_reticulations != other.max_reticulations) {
            return false;
        }
        if (configs.size() != other.configs.size()) {
            return false;
        }
        for (size_t i = 0; i < configs.size(); ++i) {
            if (configs[i].size() != other.configs[i].size()) {
                return false;
            }
            for (size_t j = 0; j < configs[i].size(); ++j) {
                if (configs[i][j] != other.configs[i][j]) {
                    return false;
                }
            }
        }

        return true;
    }

    ReticulationConfigSet() = default;

    ReticulationConfigSet(size_t max_reticulations) : max_reticulations(max_reticulations) {}

    ReticulationConfigSet(ReticulationConfigSet&& rhs) : max_reticulations{rhs.max_reticulations}
    {
        configs = std::move(rhs.configs);
    }

    ReticulationConfigSet(const ReticulationConfigSet& rhs)
      : max_reticulations{rhs.max_reticulations}
    {
        configs.clear();
        for (size_t i = 0; i < rhs.configs.size(); ++i) {
            configs.emplace_back(rhs.configs[i]);
        }
    }

    ReticulationConfigSet& operator =(ReticulationConfigSet&& rhs)
    {
        if (this != &rhs)
        {
            max_reticulations = rhs.max_reticulations;
            configs = std::move(rhs.configs);
        }
        return *this;
    }

    ReticulationConfigSet& operator =(const ReticulationConfigSet& rhs)
    {
        if (this != &rhs)
        {
            max_reticulations = rhs.max_reticulations;
            configs.clear();
            for (size_t i = 0; i < rhs.configs.size(); ++i) {
                configs.emplace_back(rhs.configs[i]);
            }
        }
        return *this;
    }
};

struct TreeLoglData {
    double tree_logprob = 0;
    bool tree_logprob_valid = false;
    bool tree_logl_valid = false;
    std::vector<double> tree_partition_logl;
    ReticulationConfigSet reticulationChoices;

    TreeLoglData(size_t n_partitions, size_t max_reticulations) : reticulationChoices(max_reticulations) {
        tree_partition_logl.resize(n_partitions);
        std::vector<ReticulationState> allChoices(max_reticulations, ReticulationState::DONT_CARE);
        reticulationChoices.configs.emplace_back(allChoices);
    }

    TreeLoglData(TreeLoglData&& rhs) : tree_logprob{rhs.tree_logprob}, tree_logprob_valid{rhs.tree_logprob_valid}, tree_logl_valid{rhs.tree_logl_valid}, tree_partition_logl{rhs.tree_partition_logl}, reticulationChoices{rhs.reticulationChoices} {}

    TreeLoglData(const TreeLoglData& rhs) : tree_logprob{rhs.tree_logprob}, tree_logprob_valid{rhs.tree_logprob_valid}, tree_logl_valid{rhs.tree_logl_valid}, tree_partition_logl{rhs.tree_partition_logl}, reticulationChoices{rhs.reticulationChoices} {}

    TreeLoglData& operator =(TreeLoglData&& rhs)
    {
        if (this != &rhs)
        {
            tree_partition_logl = rhs.tree_partition_logl;
            reticulationChoices = std::move(rhs.reticulationChoices);
            tree_logprob = rhs.tree_logprob;
            tree_logprob_valid = rhs.tree_logprob_valid;
            tree_logl_valid = rhs.tree_logl_valid;
        }
        return *this;
    }

    TreeLoglData& operator =(const TreeLoglData& rhs)
    {
        if (this != &rhs)
        {
            tree_partition_logl = rhs.tree_partition_logl;
            reticulationChoices = rhs.reticulationChoices;
            tree_logprob = rhs.tree_logprob;
            tree_logprob_valid = rhs.tree_logprob_valid;
            tree_logl_valid = rhs.tree_logl_valid;
        }
        return *this;
    }
};

struct DisplayedTreeData {
    TreeLoglData treeLoglData;
    std::vector<double*> clv_vector;
    std::vector<unsigned int*> scale_buffer;
    const std::vector<ClvRangeInfo>& clvInfo;
    const std::vector<ScaleBufferRangeInfo>& scaleBufferInfo;
    bool isTip = false;

    DisplayedTreeData(pllmod_treeinfo_t* treeinfo, const std::vector<ClvRangeInfo>& clvRangeInfo, const std::vector<ScaleBufferRangeInfo>& scaleBufferRangeInfo, size_t max_reticulations) : treeLoglData(treeinfo->partition_count, max_reticulations), clvInfo(clvRangeInfo), scaleBufferInfo(scaleBufferRangeInfo) { // inner node
        clv_vector = std::vector<double*>(treeinfo->partition_count, nullptr);
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!treeinfo->partitions[p]) {
                continue;
            }
            clv_vector[p] = create_single_empty_clv(clvRangeInfo[p]);
        }
        scale_buffer = std::vector<unsigned int*>(treeinfo->partition_count, nullptr);
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!treeinfo->partitions[p]) {
                continue;
            }
            scale_buffer[p] = create_single_empty_scale_buffer(scaleBufferRangeInfo[p]);
        }
    }

    DisplayedTreeData(pllmod_treeinfo_t* treeinfo, const std::vector<ClvRangeInfo>& clvRangeInfo, const std::vector<ScaleBufferRangeInfo>& scaleBufferRangeInfo, std::vector<double*> tip_clv_vector, size_t max_reticulations) : treeLoglData(treeinfo->partition_count, max_reticulations), clvInfo(clvRangeInfo), scaleBufferInfo(scaleBufferRangeInfo) { // tip node
        clv_vector = std::vector<double*>(treeinfo->partition_count, nullptr);
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!treeinfo->partitions[p]) {
                continue;
            }
            clv_vector[p] = tip_clv_vector[p];
        }
        isTip = true;
        scale_buffer = std::vector<unsigned int*>(treeinfo->partition_count, nullptr);
    }

    DisplayedTreeData(DisplayedTreeData&& rhs)
      : treeLoglData{rhs.treeLoglData}, clv_vector{rhs.clv_vector}, scale_buffer{rhs.scale_buffer}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}, isTip{rhs.isTip}
    {
        rhs.clv_vector.clear();
        rhs.scale_buffer.clear();
    }

    DisplayedTreeData(const DisplayedTreeData& rhs)
      : treeLoglData{rhs.treeLoglData}, clvInfo{rhs.clvInfo}, scaleBufferInfo{rhs.scaleBufferInfo}, isTip{rhs.isTip}
    {
        if (isTip) {
            clv_vector = rhs.clv_vector;
            scale_buffer = rhs.scale_buffer;
        } else {
            clv_vector = std::vector<double*>(rhs.clv_vector.size(), nullptr);
            assert(rhs.clvInfo.size() == rhs.clvInfo.size());
            for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
                clv_vector[p] = clone_single_clv_vector(rhs.clvInfo[p], rhs.clv_vector[p]);
            }
            scale_buffer = std::vector<unsigned int*>(rhs.scale_buffer.size(), nullptr);
            assert(rhs.scaleBufferInfo.size() == rhs.scale_buffer.size());
            for (size_t p = 0; p < rhs.scale_buffer.size(); ++p) {
                scale_buffer[p] = clone_single_scale_buffer(rhs.scaleBufferInfo[p], rhs.scale_buffer[p]);
            }
        }
    }

    DisplayedTreeData& operator =(DisplayedTreeData&& rhs)
    {
        if (this != &rhs)
        {
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
        }
        return *this;
    }

    DisplayedTreeData& operator =(const DisplayedTreeData& rhs)
    {
        if (this != &rhs)
        {
            treeLoglData = rhs.treeLoglData;
            if ((clv_vector.size() == rhs.clv_vector.size()) && (clvInfo == rhs.clvInfo) && (isTip == rhs.isTip)) { // simply overwrite
                if (rhs.isTip) {
                    clv_vector = rhs.clv_vector;
                } else {
                    assert(rhs.clvInfo.size() == rhs.clvInfo.size());
                    for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
                        if (!rhs.clv_vector[p]) {
                            continue;
                        }
                        assert(clv_vector[p]);
                        memcpy(clv_vector[p], rhs.clv_vector[p], rhs.clvInfo[p].inner_clv_num_entries * sizeof(double));
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
                        clv_vector = std::vector<double*>(rhs.clv_vector.size(), nullptr);
                        for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
                            clv_vector[p] = clone_single_clv_vector(rhs.clvInfo[p], rhs.clv_vector[p]);
                        }
                    }
                }
            }
            if ((scale_buffer.size() == rhs.scale_buffer.size()) && (scaleBufferInfo == rhs.scaleBufferInfo)) { // simply overwrite
                assert(rhs.scaleBufferInfo.size() == rhs.scale_buffer.size());
                for (size_t p = 0; p < rhs.scale_buffer.size(); ++p) {
                    if (!rhs.scale_buffer[p]) {
                        continue;
                    }
                    assert(scale_buffer[p]);
                    memcpy(scale_buffer[p], rhs.scale_buffer[p], rhs.scaleBufferInfo[p].scaler_size * sizeof(unsigned int));
                }
            } else {
                for (size_t p = 0; p < scale_buffer.size(); ++p) {
                    free(scale_buffer[p]);
                }
                scale_buffer = std::vector<unsigned int*>(rhs.scale_buffer.size(), nullptr);
                assert(rhs.scaleBufferInfo.size() == rhs.scale_buffer.size());
                for (size_t p = 0; p < rhs.scale_buffer.size(); ++p) {
                    scale_buffer[p] = clone_single_scale_buffer(rhs.scaleBufferInfo[p], rhs.scale_buffer[p]);
                }
            }
            isTip = rhs.isTip;
        }
        return *this;
    }

    ~DisplayedTreeData() {
        if (!isTip) {
            for (size_t p = 0; p < clv_vector.size(); ++p) {
                pll_aligned_free(clv_vector[p]);
            }
            for (size_t p = 0; p < scale_buffer.size(); ++p) {
                free(scale_buffer[p]);
            }
        }
    }
};

//double computeReticulationChoicesLogProb(const std::vector<ReticulationState>& choices, const std::vector<double>& reticulationProbs);
double computeReticulationConfigProb(const ReticulationConfigSet& choices, const std::vector<double>& reticulationProbs);
double computeReticulationConfigLogProb(const ReticulationConfigSet& choices, const std::vector<double>& reticulationProbs);

//bool reticulationChoicesCompatible(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
bool reticulationConfigsCompatible(const ReticulationConfigSet& left, const ReticulationConfigSet& right);

//void printReticulationChoices(const std::vector<ReticulationState>& reticulationChoices);
void printReticulationChoices(const ReticulationConfigSet& reticulationChoices);

//std::vector<ReticulationState> combineReticulationChoices(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
ReticulationConfigSet combineReticulationChoices(const ReticulationConfigSet& left, const ReticulationConfigSet& right);

void simplifyReticulationChoices(ReticulationConfigSet& res);

}
