#pragma once

#include "../graph/Network.hpp"
#include "../moves/Move.hpp"
#include "../optimization/NetworkState.hpp"

#include <limits>
#include <vector>
#include <deque>

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
  std::deque<PromisingState> promising_states;
};

bool addPromisingState(AnnotatedNetwork& ann_network, Move move,
                       double target_bic, PromisingStateQueue& psq);
bool hasPromisingStates(PromisingStateQueue& psq);
PromisingState getPromisingState(PromisingStateQueue& psq);

void applyPromisingState(AnnotatedNetwork& ann_network, PromisingState& pstate,
                         double* best_score, BestNetworkData* bestNetworkData,
                         bool alternative_route, bool silent);

void deleteMoveFromPSQ(AnnotatedNetwork& ann_network, PromisingStateQueue& psq, const Move& move);

}  // namespace netrax