/*
 * LikelihoodComputation.hpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <vector>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "../graph/AnnotatedNetwork.hpp"
#include "../RaxmlWrapper.hpp"
#include <raxml-ng/TreeInfo.hpp>

namespace netrax {

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental = 1, int update_pmatrices = 1);
struct SumtableInfo {
        double tree_prob = 0.0;
        double* sumtable = nullptr;
        size_t sumtable_size = 0;
        size_t alignment = 16;

        SumtableInfo(size_t sumtable_size, size_t alignment) : sumtable_size{sumtable_size}, alignment{alignment} {}

        ~SumtableInfo() {
                pll_aligned_free(sumtable);
        }

        SumtableInfo(SumtableInfo&& rhs) : tree_prob{rhs.tree_prob}, sumtable{rhs.sumtable}, sumtable_size{rhs.sumtable_size}, alignment{rhs.alignment}
        {
                rhs.sumtable = nullptr;
        }

        SumtableInfo(const SumtableInfo& rhs)
        : tree_prob{rhs.tree_prob}, sumtable_size{rhs.sumtable_size}, alignment{rhs.alignment}
        {
                sumtable = (double*) pll_aligned_alloc(rhs.sumtable_size, rhs.alignment);
                memcpy(sumtable, rhs.sumtable, rhs.sumtable_size * sizeof(double));
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
                }
                return *this;
        }

        SumtableInfo& operator =(const SumtableInfo& rhs)
        {
                if (this != &rhs)
                {
                        tree_prob = rhs.tree_prob;
                        sumtable = (double*) pll_aligned_alloc(rhs.sumtable_size, rhs.alignment);
                        memcpy(sumtable, rhs.sumtable, rhs.sumtable_size * sizeof(double));
                        sumtable_size = rhs.sumtable_size;
                        alignment = rhs.alignment;
                }
                return *this;
        }
};

struct LoglDerivatives {
        double logl_prime = std::numeric_limits<double>::infinity();
        double logl_prime_prime = std::numeric_limits<double>::infinity();
};

LoglDerivatives computeLoglikelihoodDerivatives(AnnotatedNetwork& ann_network, const std::vector<std::vector<SumtableInfo> >& sumtables, unsigned int pmatrix_index, bool incremental = true, bool update_pmatrices = true);
std::vector<std::vector<SumtableInfo> > computePartitionSumtables(AnnotatedNetwork& ann_network, unsigned int pmatrix_index);

void updateCLVsVirtualRerootTrees(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back);
double computeLoglikelihoodBrlenOpt(AnnotatedNetwork &ann_network, const std::vector<std::vector<TreeLoglData> >& oldTrees, unsigned int pmatrix_index, int incremental = 1, int update_pmatrices = 1);

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
