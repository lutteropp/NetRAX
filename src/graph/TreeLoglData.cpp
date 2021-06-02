#include "TreeLoglData.hpp"

namespace netrax {

TreeLoglData::TreeLoglData(size_t n_partitions, size_t max_reticulations) : reticulationChoices(max_reticulations) {
    tree_partition_logl.resize(n_partitions);
    std::vector<ReticulationState> allChoices(max_reticulations, ReticulationState::DONT_CARE);
    reticulationChoices.configs.emplace_back(allChoices);
}

TreeLoglData::TreeLoglData(TreeLoglData&& rhs) : tree_logprob{rhs.tree_logprob}, tree_logprob_valid{rhs.tree_logprob_valid}, tree_logl_valid{rhs.tree_logl_valid}, tree_partition_logl{rhs.tree_partition_logl}, reticulationChoices{rhs.reticulationChoices} {}

TreeLoglData::TreeLoglData(const TreeLoglData& rhs) : tree_logprob{rhs.tree_logprob}, tree_logprob_valid{rhs.tree_logprob_valid}, tree_logl_valid{rhs.tree_logl_valid}, tree_partition_logl{rhs.tree_partition_logl}, reticulationChoices{rhs.reticulationChoices} {}

TreeLoglData& TreeLoglData::operator =(TreeLoglData&& rhs)
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

TreeLoglData& TreeLoglData::operator =(const TreeLoglData& rhs)
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

}