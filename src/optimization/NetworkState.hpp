#pragma once

#include <vector>
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct EdgeInfo {
    size_t from_clv_index;
    size_t to_clv_index;
    size_t pmatrix_index;
};

struct NetworkState {
    std::vector<std::vector<double> > brlens_partitions;
    std::vector<double> reticulation_probs;
    std::vector<double> brlen_scalers;
    std::vector<EdgeInfo> edge_infos;
    std::vector<std::vector<DisplayedTreeData> > old_displayed_trees;
};

NetworkState extract_network_state(AnnotatedNetwork &ann_network);

void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state);

bool network_states_equal(NetworkState& old_state, NetworkState &act_state);

}
