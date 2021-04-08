#pragma once

#include <raxml-ng/Model.hpp>
#include <vector>
#include "../graph/AnnotatedNetwork.hpp"
#include "../likelihood/DisplayedTreeData.hpp"

namespace netrax {

struct NetworkState {
    int brlen_linkage;
    unsigned int n_trees;
    unsigned int n_branches;

    Network network;
    std::vector<std::vector<double> > partition_brlens;
    std::vector<double> partition_brlen_scalers;
    std::vector<double> alphas;
    std::vector<Model> partition_models;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    std::vector<std::vector<NodeDisplayedTreeData> > pernode_displayed_tree_data;
    std::vector<ClvRangeInfo> displayed_tree_clv_ranges;
    std::vector<ScaleBufferRangeInfo> displayed_tree_scale_buffer_ranges;

    bool network_valid = false;

    double cached_logl;
    bool cached_logl_valid;
};

bool neighborsSame(const Network& n1, const Network& n2);

NetworkState extract_network_state(AnnotatedNetwork &ann_network, bool extract_network = true);
void extract_network_state(AnnotatedNetwork &ann_network, NetworkState& state_to_reuse, bool extract_network = true);
void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state, bool copy_network = true);
bool network_states_equal(const NetworkState& old_state, const NetworkState &act_state);
AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options);

}
