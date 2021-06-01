#pragma once

#include "../graph/DisplayedTreeData.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../NetraxOptions.hpp"

namespace netrax {

struct SumtableInfo {
        double tree_prob = 0.0;
        double* sumtable = nullptr;
        size_t sumtable_size = 0;
        size_t alignment = 16;

        DisplayedTreeData* left_tree;
        DisplayedTreeData* right_tree;
        size_t left_tree_idx;
        size_t right_tree_idx;

        SumtableInfo(size_t sumtable_size, size_t alignment, DisplayedTreeData* left_tree, DisplayedTreeData* right_tree, size_t left_tree_idx, size_t right_tree_idx) : sumtable_size{sumtable_size}, alignment{alignment}, left_tree{left_tree}, right_tree{right_tree}, left_tree_idx{left_tree_idx}, right_tree_idx{right_tree_idx} {}

        ~SumtableInfo() {
                pll_aligned_free(sumtable);
        }

        SumtableInfo(SumtableInfo&& rhs) : tree_prob{rhs.tree_prob}, sumtable{rhs.sumtable}, sumtable_size{rhs.sumtable_size}, alignment{rhs.alignment}, left_tree{rhs.left_tree}, right_tree{rhs.right_tree}, left_tree_idx{rhs.left_tree_idx}, right_tree_idx{rhs.right_tree_idx}
        {
                rhs.sumtable = nullptr;
        }

        SumtableInfo& operator =(SumtableInfo&& rhs)
        {
                if (this != &rhs)
                {
                        tree_prob = rhs.tree_prob;
                        sumtable = rhs.sumtable;
                        sumtable_size = rhs.sumtable_size;
                        alignment = rhs.alignment;
                        rhs.sumtable = nullptr;
                        left_tree = rhs.left_tree;
                        right_tree = rhs.right_tree;
                        left_tree_idx = rhs.left_tree_idx;
                        right_tree_idx = rhs.right_tree_idx;
                }
                return *this;
        }

        SumtableInfo& operator =(const SumtableInfo& rhs)
        {
                if (this != &rhs)
                {
                        tree_prob = rhs.tree_prob;
                        pll_aligned_free(sumtable);
                        sumtable = (double*) pll_aligned_alloc(rhs.sumtable_size, rhs.alignment);
                        memcpy(sumtable, rhs.sumtable, rhs.sumtable_size * sizeof(double));
                        sumtable_size = rhs.sumtable_size;
                        alignment = rhs.alignment;
                        left_tree = rhs.left_tree;
                        right_tree = rhs.right_tree;
                        left_tree_idx = rhs.left_tree_idx;
                        right_tree_idx = rhs.right_tree_idx;
                }
                return *this;
        }
};

struct LoglDerivatives {
        double logl_prime = std::numeric_limits<double>::infinity();
        double logl_prime_prime = std::numeric_limits<double>::infinity();
        std::vector<double> partition_logl_prime;
        std::vector<double> partition_logl_prime_prime;
};

LoglDerivatives computeLoglikelihoodDerivatives(AnnotatedNetwork& ann_network, const std::vector<std::vector<SumtableInfo> >& sumtables, unsigned int pmatrix_index);
std::vector<std::vector<SumtableInfo> > computePartitionSumtables(AnnotatedNetwork& ann_network, unsigned int pmatrix_index);

}