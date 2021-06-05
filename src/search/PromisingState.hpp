#pragma once

#include "../graph/Network.hpp"
#include "../moves/Move.hpp"
#include "../optimization/NetworkState.hpp"

#include <limits>
#include <queue>
#include <vector>

namespace netrax {

struct AnnotatedNetwork;

struct PromisingState {
  double target_bic;
  Move move;
  Network& network;
  NetworkState state;
};

class PromisingStateComparator {
 public:
  int operator()(const PromisingState& p1, const PromisingState& p2) {
    return p1.target_bic > p2.target_bic;
  }
};

struct PromisingStateManager {
  void addPromisingState(AnnotatedNetwork& ann_network, Move move,
                         double target_bic);
  std::priority_queue<PromisingState, std::vector<PromisingState>,
                      PromisingStateComparator>
      promising_states;

 private:
  std::vector<Network> promisingNetworks;
};

void applyPromisingState(AnnotatedNetwork& ann_network, PromisingState& pstate,
                         double* best_score, BestNetworkData* bestNetworkData,
                         bool alternative_route, bool silent);

}  // namespace netrax