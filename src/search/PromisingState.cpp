#include "PromisingState.hpp"

#include "../graph/AnnotatedNetwork.hpp"
#include "Filtering.hpp"

namespace netrax {

void applyPromisingState(AnnotatedNetwork& ann_network, PromisingState& pstate,
                         double* best_score, BestNetworkData* bestNetworkData,
                         bool alternative_route, bool silent) {
  ann_network.network = pstate.network;
  if (alternative_route) {
    apply_network_state(ann_network, pstate.state,
                        true);  // we need to also update the model
  }
  acceptMove(ann_network, pstate.move, pstate.state, best_score,
             bestNetworkData, silent);
}

void PromisingStateManager::addPromisingState(AnnotatedNetwork& ann_network,
                                              Move move, double target_bic) {
  if (promisingNetworks.empty() || promisingNetworks.back() != ann_network.network) {
      promisingNetworks.emplace_back(ann_network.network);
  }
  PromisingState p{target_bic, move, promisingNetworks.back(), extract_network_state(ann_network)};
  promising_states.push(p);
}

}  // namespace netrax