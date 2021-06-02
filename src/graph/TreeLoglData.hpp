#pragma once

#include <vector>
#include "ReticulationConfigSet.hpp"

namespace netrax {
    struct TreeLoglData {
    double tree_logprob = 0;
    bool tree_logprob_valid = false;
    bool tree_logl_valid = false;
    std::vector<double> tree_partition_logl;
    ReticulationConfigSet reticulationChoices;

    TreeLoglData(size_t n_partitions, size_t max_reticulations);
    TreeLoglData(TreeLoglData&& rhs);
    TreeLoglData(const TreeLoglData& rhs);
    TreeLoglData& operator =(TreeLoglData&& rhs);
    TreeLoglData& operator =(const TreeLoglData& rhs);
};

}