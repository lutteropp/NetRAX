#pragma once

#include <raxml-ng/Model.hpp>
#include <vector>
#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/DisplayedTreeData.hpp"

namespace netrax {

struct NetworkState {
    int brlen_linkage;
    unsigned int n_trees;
    unsigned int n_branches;

    std::vector<std::vector<double> > partition_brlens; // only used in unlinked mode
    std::vector<double> linked_brlens;
    std::vector<double> partition_brlen_scalers;
    std::vector<double> alphas;
    std::vector<Model> partition_models;
    std::vector<double> reticulation_probs; // the first-parent reticulation probs

    double cached_logl;
    bool cached_logl_valid;

    NetworkState(bool) {};

    NetworkState(NetworkState&& rhs) {
        brlen_linkage = rhs.brlen_linkage;
        n_trees = rhs.n_trees;
        n_branches = rhs.n_branches;
        partition_brlens = std::move(rhs.partition_brlens);
        linked_brlens = std::move(rhs.linked_brlens);
        partition_brlen_scalers = std::move(rhs.partition_brlen_scalers);
        alphas = std::move(rhs.alphas);
        partition_models = std::move(rhs.partition_models);
        reticulation_probs = std::move(rhs.reticulation_probs);

        cached_logl = std::move(rhs.cached_logl);
        cached_logl_valid = std::move(rhs.cached_logl_valid);
    }
    NetworkState(const NetworkState& rhs) = delete;
    NetworkState& operator =(NetworkState&& rhs) {
        if (this != &rhs)
        {
            brlen_linkage = rhs.brlen_linkage;
            n_trees = rhs.n_trees;
            n_branches = rhs.n_branches;
            partition_brlens = std::move(rhs.partition_brlens);
            linked_brlens = std::move(rhs.linked_brlens);
            partition_brlen_scalers = std::move(rhs.partition_brlen_scalers);
            alphas = std::move(rhs.alphas);
            partition_models = std::move(rhs.partition_models);
            reticulation_probs = std::move(rhs.reticulation_probs);
;
            cached_logl = std::move(rhs.cached_logl);
            cached_logl_valid = std::move(rhs.cached_logl_valid);
        }
        return *this;
    }
    NetworkState& operator =(const NetworkState& rhs) = delete;
};

bool neighborsSame(const Network& n1, const Network& n2);

NetworkState extract_network_state(AnnotatedNetwork &ann_network);
void extract_network_state(AnnotatedNetwork &ann_network, NetworkState& state_to_reuse);
void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state);
bool network_states_equal(const NetworkState& old_state, const NetworkState &act_state);
AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options);

}
