#pragma once

#include <raxml-ng/Model.hpp>
#include <vector>
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct NetworkState {
    int brlen_linkage;
    Network network;
    std::vector<std::vector<double> > partition_brlens;
    std::vector<double> partition_brlen_scalers;
    std::vector<Model> partition_models;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    bool network_valid = false;
};

NetworkState extract_network_state(AnnotatedNetwork &ann_network, bool extract_network = true);
void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state, bool copy_network = true);
bool network_states_equal(const NetworkState& old_state, const NetworkState &act_state);
AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options);

}