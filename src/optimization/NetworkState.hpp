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
    std::vector<std::vector<double> > partition_brlens; // only used in unlinked mode
    std::vector<double> linked_brlens;
    std::vector<double> partition_brlen_scalers;
    std::vector<double> alphas;
    std::vector<Model> partition_models;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    std::vector<double**> partition_pmatrix; // per-partition pmatrices

    std::vector<NodeDisplayedTreeData> pernode_displayed_tree_data;
    std::vector<ClvRangeInfo> displayed_tree_clv_ranges;
    std::vector<ScaleBufferRangeInfo> displayed_tree_scale_buffer_ranges;

    bool network_valid = false;

    double cached_logl;
    bool cached_logl_valid;

    ~NetworkState() {
        for (size_t p = 0; p < partition_pmatrix.size(); ++p) {
            if (partition_pmatrix[p]) {
                pll_aligned_free(partition_pmatrix[p][0]);
            }
            free(partition_pmatrix[p]);
        }
    }

    NetworkState() = default;

    NetworkState(NetworkState&& rhs) {
        brlen_linkage = rhs.brlen_linkage;
        n_trees = rhs.n_trees;
        n_branches = rhs.n_branches;
        network = std::move(rhs.network);
        partition_brlens = std::move(rhs.partition_brlens);
        linked_brlens = std::move(rhs.linked_brlens);
        partition_brlen_scalers = std::move(rhs.partition_brlen_scalers);
        alphas = std::move(rhs.alphas);
        partition_models = std::move(rhs.partition_models);
        reticulation_probs = std::move(rhs.reticulation_probs);

        partition_pmatrix = std::move(rhs.partition_pmatrix);
        pernode_displayed_tree_data = std::move(rhs.pernode_displayed_tree_data);
        displayed_tree_clv_ranges = std::move(rhs.displayed_tree_clv_ranges);
        displayed_tree_scale_buffer_ranges = std::move(rhs.displayed_tree_scale_buffer_ranges);
        network_valid = std::move(rhs.network_valid);
        cached_logl = std::move(rhs.cached_logl);
        cached_logl_valid = std::move(rhs.cached_logl_valid);

        for (size_t p = 0; p < rhs.partition_pmatrix.size(); ++p) {
            rhs.partition_pmatrix[p] = nullptr;
        }
    }
    NetworkState(const NetworkState& rhs) = delete;
    NetworkState& operator =(NetworkState&& rhs) {
        if (this != &rhs)
        {
            brlen_linkage = rhs.brlen_linkage;
            n_trees = rhs.n_trees;
            n_branches = rhs.n_branches;
            network = std::move(rhs.network);
            partition_brlens = std::move(rhs.partition_brlens);
            linked_brlens = std::move(rhs.linked_brlens);
            partition_brlen_scalers = std::move(rhs.partition_brlen_scalers);
            alphas = std::move(rhs.alphas);
            partition_models = std::move(rhs.partition_models);
            reticulation_probs = std::move(rhs.reticulation_probs);

            partition_pmatrix = std::move(rhs.partition_pmatrix);
            pernode_displayed_tree_data = std::move(rhs.pernode_displayed_tree_data);
            displayed_tree_clv_ranges = std::move(rhs.displayed_tree_clv_ranges);
            displayed_tree_scale_buffer_ranges = std::move(rhs.displayed_tree_scale_buffer_ranges);
            network_valid = std::move(rhs.network_valid);
            cached_logl = std::move(rhs.cached_logl);
            cached_logl_valid = std::move(rhs.cached_logl_valid);

            for (size_t p = 0; p < rhs.partition_pmatrix.size(); ++p) {
                rhs.partition_pmatrix[p] = nullptr;
            }
        }
        return *this;
    }
    NetworkState& operator =(const NetworkState& rhs) = delete;
};

bool neighborsSame(const Network& n1, const Network& n2);

NetworkState extract_network_state(AnnotatedNetwork &ann_network, bool extract_network = true);
void extract_network_state(AnnotatedNetwork &ann_network, NetworkState& state_to_reuse, bool extract_network = true);
void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state, bool copy_network = true);
bool network_states_equal(const NetworkState& old_state, const NetworkState &act_state);
AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options);

}
