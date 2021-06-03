#pragma once

#include <limits>
#include <string>
#include <vector>
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct ScoreImprovementResult {
  bool local_improved = false;
  bool global_improved = false;
};

struct BestNetworkData {
  size_t best_n_reticulations = 0;
  std::vector<double> logl;
  std::vector<double> bic;
  std::vector<std::string> newick;
  BestNetworkData(size_t max_reticulations) {
    logl = std::vector<double>(max_reticulations + 1,
                               -std::numeric_limits<double>::infinity());
    bic.resize(max_reticulations + 1, std::numeric_limits<double>::infinity());
    newick.resize(max_reticulations + 1, "");
  }
};

ScoreImprovementResult check_score_improvement(AnnotatedNetwork &ann_network,
                                               double *local_best,
                                               BestNetworkData *bestNetworkData,
                                               bool silent = false);

}  // namespace netrax