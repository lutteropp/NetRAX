#pragma once

#include <raxml-ng/Model.hpp>
#include <vector>
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct NetworkState {
    Network network;
    std::vector<std::vector<double> > partition_brlens;
    std::vector<Model> partition_models;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs
};

NetworkState extract_network_state(AnnotatedNetwork &ann_network);
void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state);
bool network_states_equal(const NetworkState& old_state, const NetworkState &act_state);
AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options);

}