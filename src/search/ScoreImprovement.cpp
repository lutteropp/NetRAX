#include "ScoreImprovement.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../NetraxOptions.hpp"
#include "../colormod.h"  // namespace Color
#include "../io/NetworkIO.hpp"
#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"

namespace netrax {

ScoreImprovementResult check_score_improvement(AnnotatedNetwork &ann_network,
                                               double *local_best,
                                               BestNetworkData *bestNetworkData,
                                               bool silent) {
  bool local_improved = false;
  bool global_improved = false;

  double new_score = scoreNetwork(ann_network);
  double score_diff =
      bestNetworkData->bic[ann_network.network.num_reticulations()] - new_score;
  if (score_diff > 0) {
    double old_global_best =
        bestNetworkData->bic[bestNetworkData->best_n_reticulations];
    bestNetworkData->bic[ann_network.network.num_reticulations()] = new_score;
    bestNetworkData->logl[ann_network.network.num_reticulations()] =
        computeLoglikelihood(ann_network, 1, 1);
    bestNetworkData->newick[ann_network.network.num_reticulations()] =
        toExtendedNewick(ann_network);

    local_improved = true;

    if (new_score < old_global_best) {
      bestNetworkData->best_n_reticulations =
          ann_network.network.num_reticulations();
      global_improved = true;
      if (ann_network.fake_treeinfo->brlen_linkage ==
          PLLMOD_COMMON_BRLEN_UNLINKED) {
        collect_average_branches(ann_network);
      }
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << BOLDGREEN;
        std::cout << "IMPROVED GLOBAL BEST SCORE FOUND SO FAR ("
                  << ann_network.network.num_reticulations()
                  << " reticulations): " << new_score << "\n";
        std::cout << RESET;
        writeNetwork(ann_network, ann_network.options.output_file);
        if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
        if (!silent)
          std::cout << "Better network written to "
                    << ann_network.options.output_file << "\n";
      }
      *local_best = new_score;
    } else if (new_score < *local_best) {
      *local_best = new_score;
    }
  }
  return ScoreImprovementResult{local_improved, global_improved};
}

}  // namespace netrax