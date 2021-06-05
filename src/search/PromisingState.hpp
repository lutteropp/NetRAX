#pragma once

#include "../graph/Network.hpp"
#include "../moves/Move.hpp"
#include "../optimization/NetworkState.hpp"

#include <limits>
#include <unordered_set>
#include <vector>

namespace netrax {

struct AnnotatedNetwork;
struct BestNetworkData;

struct PromisingState {
  double target_bic;
  Move move;
  Network network;
  NetworkState state;
};

class PromisingStateComparator {
 public:
  int operator()(const PromisingState& p1, const PromisingState& p2) {
    return p1.target_bic > p2.target_bic;
  }
};

struct PromisingStateQueue {
  std::vector<PromisingState> promising_states;
};

void addPromisingState(AnnotatedNetwork& ann_network, Move move,
                       double target_bic, PromisingStateQueue& psq);
bool hasPromisingStates(PromisingStateQueue& psq);
PromisingState getPromisingState(PromisingStateQueue& psq);

void applyPromisingState(AnnotatedNetwork& ann_network, PromisingState& pstate,
                         double* best_score, BestNetworkData* bestNetworkData,
                         bool alternative_route, bool silent);

}  // namespace netrax