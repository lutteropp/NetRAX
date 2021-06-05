#include "PromisingState.hpp"

#include <algorithm>
#include <stdexcept>

#include "../DebugPrintFunctions.hpp"
#include "../NetraxOptions.hpp"
#include "../graph/AnnotatedNetwork.hpp"
#include "../helper/NetworkFunctions.hpp"
#include "Filtering.hpp"

namespace netrax {

PromisingState extractPromisingState(AnnotatedNetwork& ann_network, Move move,
                                     double target_bic) {
  PromisingState p{target_bic, move, ann_network.network,
                   extract_network_state(ann_network)};
  return p;
}

void applyPromisingState(AnnotatedNetwork& ann_network, PromisingState& pstate,
                         double* best_score, BestNetworkData* bestNetworkData,
                         bool alternative_route, bool silent) {
  if (ann_network.options.retry == 0) {
    return;
  }
  ann_network.network = std::move(pstate.network);
  if (alternative_route) {
    apply_network_state(ann_network, pstate.state,
                        true);  // we need to also update the model
  }
  ann_network.travbuffer = reversed_topological_sort(ann_network.network);
  acceptMove(ann_network, pstate.move, nullptr, best_score, bestNetworkData,
             true);
}

bool addPromisingState(AnnotatedNetwork& ann_network, Move move,
                       double target_bic, PromisingStateQueue& psq) {
  if (ann_network.options.retry == 0) {
    return false;
  }
  size_t max_entries =
      (ann_network.options.retry == 0) ? 0 : ann_network.options.retry * 4;

  // We do not have to store more good states in the PSQ than
  // ann_network.options.retry. Just keep the best ones.
  if (psq.promising_states.size() > max_entries) {
    std::sort(psq.promising_states.begin(), psq.promising_states.end(),
              [](const PromisingState& p1, const PromisingState& p2) {
                return p1.target_bic > p2.target_bic;
              });
  }
  while (psq.promising_states.size() > max_entries) {
    if (psq.promising_states.back().target_bic > target_bic) {
      psq.promising_states.pop_back();
    }
  }

  if (psq.promising_states.size() < max_entries) {
    // TODO: Take care of duplicates. Otherwise, candidates will be added
    // multiple times.
    PromisingState p = extractPromisingState(ann_network, move, target_bic);
    psq.promising_states.emplace_back(p);
    return true;
  }
  return false;
}

bool hasPromisingStates(PromisingStateQueue& psq) {
  return (!psq.promising_states.empty());
}

PromisingState getPromisingState(PromisingStateQueue& psq) {
  std::sort(psq.promising_states.begin(), psq.promising_states.end(),
            [](const PromisingState& p1, const PromisingState& p2) {
              return p1.target_bic > p2.target_bic;
            });
  if (!psq.promising_states.empty()) {
    PromisingState res = std::move(psq.promising_states.back());
    psq.promising_states.pop_back();
    return res;
  }
  throw std::runtime_error("Promising state queue is empty");
}

void deleteMoveFromPSQ(AnnotatedNetwork& ann_network, PromisingStateQueue& psq,
                       const Move& move) {
  if (ann_network.options.retry == 0) {
    return;
  }
  std::string networkInfo = exportDebugInfoNetwork(ann_network.network);
  psq.promising_states.erase(
      std::remove_if(psq.promising_states.begin(), psq.promising_states.end(),
                     [&move, &networkInfo](PromisingState& ps) {
                       return (ps.move == move &&
                               networkInfo ==
                                   exportDebugInfoNetwork(ps.network));
                     }),
      psq.promising_states.end());
}

}  // namespace netrax
