/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

#include "mpreal.h"

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "../graph/AnnotatedNetwork.hpp"
#include "../RaxmlWrapper.hpp"
#include <raxml-ng/TreeInfo.hpp>

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

std::vector<DisplayedTreeData> extractOldTrees(AnnotatedNetwork& ann_network, Node* virtual_root);

double computePseudoLoglikelihood(AnnotatedNetwork& ann_network, int incremental = 1, int update_pmatrices = 1);
double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental = 1, int update_pmatrices = 1);

LoglDerivatives computeLoglikelihoodDerivativesPseudo(AnnotatedNetwork& ann_network, const std::vector<double*>& sumtables, unsigned int pmatrix_index);
LoglDerivatives computeLoglikelihoodDerivatives(AnnotatedNetwork& ann_network, const std::vector<std::vector<SumtableInfo> >& sumtables, const std::vector<DisplayedTreeData>& oldTree, unsigned int pmatrix_index);
std::vector<double*> computePartitionSumtablesPseudo(AnnotatedNetwork& ann_network, unsigned int pmatrix_index);
std::vector<std::vector<SumtableInfo> > computePartitionSumtables(AnnotatedNetwork& ann_network, unsigned int pmatrix_index);
//double computeLoglikelihoodFromSumtables(AnnotatedNetwork& ann_network, const std::vector<std::vector<SumtableInfo> >& sumtables, const std::vector<std::vector<TreeLoglData> >& oldTrees, unsigned int pmatrix_index, bool incremental = true, bool update_pmatrices = true);

void updateCLVsVirtualRerootTrees(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back);
double computeLoglikelihoodBrlenOpt(AnnotatedNetwork &ann_network, const std::vector<DisplayedTreeData>& oldTrees, unsigned int pmatrix_index, int incremental = 1, int update_pmatrices = 1);

double computeLoglikelihoodNaiveUtree(AnnotatedNetwork &ann_network, int incremental,
        int update_pmatrices, std::vector<double> *treewise_logl = nullptr);

void setup_pmatrices(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices);
double displayed_tree_logprob(AnnotatedNetwork &ann_network, size_t tree_index);

size_t get_param_count(AnnotatedNetwork& ann_network);
size_t get_sample_size(AnnotatedNetwork& ann_network);
double aic(AnnotatedNetwork &ann_network, double logl);
double aicc(AnnotatedNetwork &ann_network, double logl);
double bic(AnnotatedNetwork &ann_network, double logl);
double scoreNetwork(AnnotatedNetwork &ann_network);

}
