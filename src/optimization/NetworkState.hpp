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

    std::vector<std::vector<double**> > displayed_tree_clv_data;
    std::vector<ClvRangeInfo> displayed_tree_clv_ranges;
    std::vector<std::vector<unsigned int**> > displayed_tree_scale_buffer_data;
    std::vector<ScaleBufferRangeInfo> displayed_tree_scale_buffer_ranges;

    bool network_valid = false;

    NetworkState& operator=(NetworkState&& other) = default;
    NetworkState(NetworkState&& other) = default;
    NetworkState() = default;
    ~NetworkState() {
        for (size_t i = 0; i < displayed_tree_clv_data.size(); ++i) {
            for (size_t j = 0; j < displayed_tree_clv_data[i].size(); ++j) {
                delete_cloned_clv_vector(displayed_tree_clv_ranges[i], displayed_tree_clv_data[i][j]);
                delete_cloned_scale_buffer(displayed_tree_scale_buffer_ranges[i], displayed_tree_scale_buffer_data[i][j]);
            }
        }
    }
};

NetworkState extract_network_state(AnnotatedNetwork &ann_network, bool extract_network = true);
void extract_network_state(AnnotatedNetwork &ann_network, NetworkState& state_to_reuse, bool extract_network = true);
void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state, bool copy_network = true);
bool network_states_equal(const NetworkState& old_state, const NetworkState &act_state);
AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options);

}