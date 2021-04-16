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

    ClvRangeInfo() = default;

    ClvRangeInfo(unsigned int alignment, unsigned int total_num_clvs, unsigned int start, unsigned int end, size_t inner_clv_num_entries) : alignment{alignment}, total_num_clvs{total_num_clvs}, start{start}, end{end}, inner_clv_num_entries{inner_clv_num_entries} {}

    ClvRangeInfo(ClvRangeInfo&& rhs) : alignment{rhs.alignment}, total_num_clvs{rhs.total_num_clvs}, start{rhs.start}, end{rhs.end}, inner_clv_num_entries{rhs.inner_clv_num_entries} {}

    ClvRangeInfo(const ClvRangeInfo& rhs) : alignment{rhs.alignment}, total_num_clvs{rhs.total_num_clvs}, start{rhs.start}, end{rhs.end}, inner_clv_num_entries{rhs.inner_clv_num_entries} {}

    ClvRangeInfo& operator =(ClvRangeInfo&& rhs)
    {
        if (this != &rhs)
        {
            alignment = rhs.alignment;
            total_num_clvs = rhs.total_num_clvs;
            start = rhs.start;
            end = rhs.end;
            inner_clv_num_entries = rhs.inner_clv_num_entries;
        }
        return *this;
    }

    ClvRangeInfo& operator =(const ClvRangeInfo& rhs)
    {
        if (this != &rhs)
        {
            alignment = rhs.alignment;
            total_num_clvs = rhs.total_num_clvs;
            start = rhs.start;
            end = rhs.end;
            inner_clv_num_entries = rhs.inner_clv_num_entries;
        }
        return *this;
    }
};

struct ScaleBufferRangeInfo {
    unsigned int scaler_size = 0;
    unsigned int num_scale_buffers = 0;
    bool operator==(const ScaleBufferRangeInfo& other) const {
        return ((scaler_size == other.scaler_size) && (num_scale_buffers == other.num_scale_buffers));
    }

    ScaleBufferRangeInfo() = default;

    ScaleBufferRangeInfo(unsigned int scaler_size, unsigned int num_scale_buffers) : scaler_size{scaler_size}, num_scale_buffers{num_scale_buffers} {}

    ScaleBufferRangeInfo(ScaleBufferRangeInfo&& rhs) : scaler_size{rhs.scaler_size}, num_scale_buffers{rhs.num_scale_buffers} {}

    ScaleBufferRangeInfo(const ScaleBufferRangeInfo& rhs) : scaler_size{rhs.scaler_size}, num_scale_buffers{rhs.num_scale_buffers} {}

    ScaleBufferRangeInfo& operator =(ScaleBufferRangeInfo&& rhs)
    {
        if (this != &rhs)
        {
            scaler_size = rhs.scaler_size;
            num_scale_buffers = rhs.num_scale_buffers;
        }
        return *this;
    }

    ScaleBufferRangeInfo& operator =(const ScaleBufferRangeInfo& rhs)
    {
        if (this != &rhs)
        {
            scaler_size = rhs.scaler_size;
            num_scale_buffers = rhs.num_scale_buffers;
        }
        return *this;
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
    TAKE_SECOND_PARENT = 2,
    INVALID = 3
};

struct ReticulationConfigSet {
    std::vector<std::vector<ReticulationState> > configs;
    size_t max_reticulations = 0;

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
    std::vector<bool> tree_logl_valid;
    bool tree_logprob_valid = false;
    std::vector<double> tree_logl;
    double tree_logprob;
    ReticulationConfigSet reticulationChoices;

    TreeLoglData(size_t n_partitions) {
        tree_logl_valid = std::vector<bool>(n_partitions, false);
        tree_logl = std::vector<double>(n_partitions, 0.0);
    }

    TreeLoglData(size_t n_partitions, size_t max_reticulations) : reticulationChoices(max_reticulations) {
        tree_logl_valid = std::vector<bool>(n_partitions, false);
        tree_logl = std::vector<double>(n_partitions, 0.0);
        std::vector<ReticulationState> allChoices(max_reticulations, ReticulationState::DONT_CARE);
        reticulationChoices.configs.emplace_back(allChoices);
    }

    TreeLoglData(TreeLoglData&& rhs) : tree_logl_valid{rhs.tree_logl_valid}, tree_logprob_valid{rhs.tree_logprob_valid}, tree_logl{rhs.tree_logl}, tree_logprob{rhs.tree_logprob}, reticulationChoices{rhs.reticulationChoices} {}

    TreeLoglData(const TreeLoglData& rhs) : tree_logl_valid{rhs.tree_logl_valid}, tree_logprob_valid{rhs.tree_logprob_valid}, tree_logl{rhs.tree_logl}, tree_logprob{rhs.tree_logprob}, reticulationChoices{rhs.reticulationChoices} {}

    TreeLoglData& operator =(TreeLoglData&& rhs)
    {
        if (this != &rhs)
        {
            tree_logl_valid = rhs.tree_logl_valid;
            tree_logprob_valid = rhs.tree_logprob_valid;
            tree_logl = rhs.tree_logl;
            tree_logprob = rhs.tree_logprob;
            reticulationChoices = std::move(rhs.reticulationChoices);
        }
        return *this;
    }

    TreeLoglData& operator =(const TreeLoglData& rhs)
    {
        if (this != &rhs)
        {
            tree_logl_valid = rhs.tree_logl_valid;
            tree_logprob_valid = rhs.tree_logprob_valid;
            tree_logl = rhs.tree_logl;
            tree_logprob = rhs.tree_logprob;
            reticulationChoices = rhs.reticulationChoices;
        }
        return *this;
    }
};

struct DisplayedTreeData {
    TreeLoglData treeLoglData;

    std::vector<double*> clv_vector;
    std::vector<unsigned int*> scale_buffer;
    std::vector<ClvRangeInfo> clvInfo;
    std::vector<ScaleBufferRangeInfo> scaleBufferInfo;
    bool isTip = false;

    DisplayedTreeData(pllmod_treeinfo_t* treeinfo, const std::vector<ClvRangeInfo>& clvRangeInfo, const std::vector<ScaleBufferRangeInfo>& scaleBufferRangeInfo, size_t max_reticulations) : treeLoglData(max_reticulations) { // inner node
        clv_vector = std::vector<double*>(treeinfo->partition_count, nullptr);
        scale_buffer = std::vector<unsigned int*>(treeinfo->partition_count, nullptr);
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            if (treeinfo->partitions[p]) {
                clv_vector[p] = create_single_empty_clv(clvRangeInfo[p]);
                scale_buffer[p] = create_single_empty_scale_buffer(scaleBufferRangeInfo[p]);
            }
        }
        
        this->clvInfo = clvRangeInfo;
        this->scaleBufferInfo = scaleBufferRangeInfo;
    }

    DisplayedTreeData(pllmod_treeinfo_t* treeinfo, std::vector<double*> tip_clv_vector, size_t max_reticulations) : treeLoglData(max_reticulations) { // tip node
        clv_vector = std::vector<double*>(treeinfo->partition_count, nullptr);
        scale_buffer = std::vector<unsigned int*>(treeinfo->partition_count, nullptr);
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            if (treeinfo->partitions[p]) {
                clv_vector[p] = tip_clv_vector[p];
            }
        }
        
        isTip = true;
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
            scale_buffer = std::vector<unsigned int*>(rhs.scale_buffer.size(), nullptr);
            for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
                clv_vector[p] = clone_single_clv_vector(rhs.clvInfo[p], rhs.clv_vector[p]);
            }
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
                pll_aligned_free(scale_buffer[p]);
            }
            treeLoglData = rhs.treeLoglData;
            clv_vector = rhs.clv_vector;
            scale_buffer = rhs.scale_buffer;
            clvInfo = rhs.clvInfo;
            scaleBufferInfo = rhs.scaleBufferInfo;

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
            if ((!clv_vector.empty() && !rhs.clv_vector.empty()) && (clvInfo == rhs.clvInfo) && (isTip == rhs.isTip)) { // simply overwrite
                if (rhs.isTip) {
                    clv_vector = rhs.clv_vector;
                } else {
                    for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
                        memcpy(clv_vector[p], rhs.clv_vector[p], clvInfo[p].inner_clv_num_entries * sizeof(double));
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
                        clv_vector = std::vector<double*>(rhs.clv_vector.size(), nullptr);
                        for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
                            clv_vector[p] = clone_single_clv_vector(rhs.clvInfo[p], rhs.clv_vector[p]);
                        }
                    }
                }
            }
            if ((!scale_buffer.empty() && !rhs.scale_buffer.empty()) && (scaleBufferInfo == rhs.scaleBufferInfo)) { // simply overwrite
                for (size_t p = 0; p < rhs.clv_vector.size(); ++p) {
                    memcpy(scale_buffer[p], rhs.scale_buffer[p], scaleBufferInfo[p].scaler_size * sizeof(unsigned int));
                }
            } else {
                for (size_t p = 0; p < scale_buffer.size(); ++p) {
                    free(scale_buffer[p]);
                }
                scale_buffer = std::vector<unsigned int*>(rhs.scale_buffer.size(), nullptr);
                for (size_t p = 0; p < rhs.scale_buffer.size(); ++p) {
                    scale_buffer[p] = clone_single_scale_buffer(rhs.scaleBufferInfo[p], rhs.scale_buffer[p]);
                }
            }
            clvInfo = rhs.clvInfo;
            scaleBufferInfo = rhs.scaleBufferInfo;
            isTip = rhs.isTip;
        }
        return *this;
    }

    ~DisplayedTreeData() {
        if (!isTip) {
            for (size_t p = 0; p < clv_vector.size(); ++p) {
                pll_aligned_free(clv_vector[p]);
            }
        }
        for (size_t p = 0; p < scale_buffer.size(); ++p) {
            free(scale_buffer[p]);
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