#include "PromisingState.hpp"

#include <stdexcept>

#include "../graph/AnnotatedNetwork.hpp"
#include "Filtering.hpp"
#include "../NetraxOptions.hpp"
#include "../helper/NetworkFunctions.hpp"

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
  ann_network.network = std::move(pstate.network);
  if (alternative_route) {
    apply_network_state(ann_network, pstate.state,
                        true);  // we need to also update the model
  }
  ann_network.travbuffer = reversed_topological_sort(ann_network.network);
  acceptMove(ann_network, pstate.move, pstate.state, best_score,
             bestNetworkData, silent);
}

void addPromisingState(AnnotatedNetwork& ann_network, Move move,
                       double target_bic, PromisingStateQueue& psq) {
  // We do not have to store more good states in the PSQ than ann_network.options.retry. Just keep the best ones.
  while (psq.promising_states.size() > ann_network.options.retry) {
      if (psq.promising_states.back().target_bic > target_bic) {
          psq.promising_states.pop_back();
      }
  }

  if (psq.promising_states.size() < ann_network.options.retry) {
      PromisingState p = extractPromisingState(ann_network, move, target_bic);
      psq.promising_states.emplace_back(p);
  }

  // TODO: Take care of duplicates. Otherwise, candidates will be added multiple times.
}

bool hasPromisingStates(PromisingStateQueue& psq) {
  return (!psq.promising_states.empty());
}

PromisingState getPromisingState(PromisingStateQueue& psq) {
  std::sort(psq.promising_states.begin(), psq.promising_states.end(), [](const PromisingState& p1, const PromisingState& p2) {return p1.target_bic > p2.target_bic;});
  if (!psq.promising_states.empty()) {
    PromisingState res = std::move(psq.promising_states.back());
    psq.promising_states.pop_back();
    return res;
  }
  throw std::runtime_error("Promising state queue is empty");
}

}  // namespace netrax