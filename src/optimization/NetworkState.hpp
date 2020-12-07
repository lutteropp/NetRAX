#pragma once

#include <vector>
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct NetworkState {
    std::vector<std::vector<double> > brlens_partitions;
    std::vector<double> brlens_network;
    std::vector<double> brprobs;
    std::vector<double> brlen_scalers;
};

NetworkState extract_network_state(AnnotatedNetwork &ann_network);

void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state);

bool network_states_equal(NetworkState& old_state, NetworkState &act_state);

}