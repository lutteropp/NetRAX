#include "PromisingState.hpp"

#include <stdexcept>

#include "../graph/AnnotatedNetwork.hpp"
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
  ann_network.network = std::move(pstate.network);
  if (alternative_route) {
    apply_network_state(ann_network, pstate.state,
                        true);  // we need to also update the model
  }
  acceptMove(ann_network, pstate.move, pstate.state, best_score,
             bestNetworkData, silent);
}

void addPromisingState(AnnotatedNetwork& ann_network, Move move,
                       double target_bic, PromisingStateQueue& psq) {
  PromisingState p = extractPromisingState(ann_network, move, target_bic);
  psq.promising_states.emplace(p);
  // TODO: Take care of duplicates. Otherwise, candidates will be added multiple times.
}

bool hasPromisingStates(PromisingStateQueue& psq) {
  while (!psq.promising_states.empty()) {
    const PromisingState& p = psq.promising_states.top();
    if (psq.deleted_elements.count(&p) != 0) {
      psq.promising_states.pop();
    } else {
      return true;
    }
  }
  return false;
}

PromisingState getPromisingState(PromisingStateQueue& psq) {
  while (!psq.promising_states.empty()) {
    const PromisingState& p = psq.promising_states.top();
    if (psq.deleted_elements.count(&p) != 0) {
      psq.promising_states.pop();
    } else {
      psq.deleted_elements.emplace(&p);
      return p;
    }
  }
  throw std::runtime_error("Promising state queue is empty");
}

}  // namespace netrax