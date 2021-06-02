#include "SimulatedAnnealing.hpp"

#include "../likelihood/ComplexityScoring.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../moves/Move.hpp"
#include "../optimization/BranchLengthOptimization.hpp"
#include "../optimization/Optimization.hpp"
#include "CandidateSelection.hpp"

#include "../DebugPrintFunctions.hpp"
#include "../io/NetworkIO.hpp"

namespace netrax {

template <typename T>
bool simanneal_step(AnnotatedNetwork &ann_network, double t,
                    std::vector<T> neighbors, const NetworkState &oldState,
                    std::unordered_set<double> &seen_bics, bool silent = true) {
  if (neighbors.empty() || t <= 0) {
    return false;
  }

  if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
    if (!silent)
      std::cout << "MoveType: " << toString(neighbors[0].moveType) << "\n";
    if (!silent) std::cout << "t: " << t << "\n";
  }

  double brlen_smooth_factor = 0.25;
  int max_iters = brlen_smooth_factor * RAXML_BRLEN_SMOOTHINGS;
  int max_iters_outside = max_iters;
  int radius = 1;

  double old_bic = scoreNetwork(ann_network);

  for (size_t i = 0; i < neighbors.size(); ++i) {
    T move(neighbors[i]);
    assert(checkSanity(ann_network, move));
    performMove(ann_network, move);
    std::unordered_set<size_t> brlen_opt_candidates =
        brlenOptCandidates(ann_network, move);
    assert(!brlen_opt_candidates.empty());
    add_neighbors_in_radius(ann_network, brlen_opt_candidates, 1);
    optimize_branches(ann_network, max_iters, max_iters_outside, radius,
                      brlen_opt_candidates);
    optimizeReticulationProbs(ann_network);

    double bicScore = scoreNetwork(ann_network);

    if (bicScore < old_bic) {
      if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        if (!silent) std::cout << " Took " << toString(move.moveType) << "\n";
        if (!silent)
          std::cout << "  Logl: " << computeLoglikelihood(ann_network)
                    << ", BIC: " << scoreNetwork(ann_network) << "\n";
        if (!silent)
          std::cout << "  num_reticulations: "
                    << ann_network.network.num_reticulations() << "\n";
        if (!silent) std::cout << toExtendedNewick(ann_network) << "\n";
      }
      return true;
    }

    if (seen_bics.count(bicScore) == 0) {
      seen_bics.emplace(bicScore);
      double acceptance_ratio =
          exp(-((bicScore - old_bic) /
                t));  // I took this one from:
                      // https://de.wikipedia.org/wiki/Simulated_Annealing
      double x = std::uniform_real_distribution<double>(0, 1)(ann_network.rng);
      if (x <= acceptance_ratio) {
        if (ParallelContext::master_rank() &&
            ParallelContext::master_thread()) {
          if (!silent) std::cout << " Took " << toString(move.moveType) << "\n";
          if (!silent)
            std::cout << "  Logl: " << computeLoglikelihood(ann_network)
                      << ", BIC: " << scoreNetwork(ann_network) << "\n";
          if (!silent)
            std::cout << "  num_reticulations: "
                      << ann_network.network.num_reticulations() << "\n";
        }
        return true;
      }
    }
    apply_network_state(ann_network, oldState);
    assert(checkSanity(ann_network, neighbors[i]));
  }

  return false;
}

double update_temperature(double t) {
  return t *
         0.95;  // TODO: Better temperature update ? I took this one from:
                // https://de.mathworks.com/help/gads/how-simulated-annealing-works.html
}

double simanneal(AnnotatedNetwork &ann_network, double t_start,
                 bool rspr1_present, bool delta_plus_present, MoveType type,
                 NetworkState &start_state_to_reuse,
                 NetworkState &best_state_to_reuse,
                 BestNetworkData *bestNetworkData, bool silent) {
  double start_bic = scoreNetwork(ann_network);
  double best_bic = start_bic;
  extract_network_state(ann_network, best_state_to_reuse);
  extract_network_state(ann_network, start_state_to_reuse);
  double t = t_start;
  bool network_changed = true;
  std::unordered_set<double> seen_bics;

  while (network_changed) {
    network_changed = false;
    extract_network_state(ann_network, start_state_to_reuse);

    network_changed = simanneal_step(
        ann_network, t,
        possibleMoves(ann_network, type, rspr1_present, delta_plus_present),
        start_state_to_reuse, seen_bics, silent);

    if (network_changed) {
      double act_bic = scoreNetwork(ann_network);
      if (act_bic < best_bic) {
        optimizeAllNonTopology(ann_network, OptimizeAllNonTopologyType::NORMAL);
        check_score_improvement(ann_network, &best_bic, bestNetworkData);
        extract_network_state(ann_network, best_state_to_reuse);
      }
    }

    t = update_temperature(t);
  }

  apply_network_state(ann_network, best_state_to_reuse);
  return computeLoglikelihood(ann_network);
}

}  // namespace netrax