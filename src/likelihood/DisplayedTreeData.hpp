#include <vector>

namespace netrax {

struct DisplayedTreeData {
    size_t tree_idx = 0;
    double tree_logl = 0.0;
    double tree_logprob = 0.0;
    std::vector<double> tree_clv;
    std::vector<double> tree_persite_logl;
};

}