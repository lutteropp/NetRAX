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
    std::vector<double> tree_clv;
    std::vector<double> tree_persite_logl;
};

double** clone_clv_vector(pll_partition_t* partition);
void delete_cloned_clv_vector(pll_partition_t* partition, double** clv);

}