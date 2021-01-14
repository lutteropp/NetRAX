#include "NetworkState.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../likelihood/LikelihoodComputation.hpp"

namespace netrax {

void assert_tip_links(const Network& network) {
    for (size_t i = 0; i < network.num_tips(); ++i) {
        assert(network.nodes_by_index[i]->clv_index == i);
        assert(!network.nodes_by_index[i]->label.empty());
        assert(network.nodes_by_index[i]->links.size() == 1);
    }
}

bool consecutive_indices(const Network& network) {
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        assert(network.nodes_by_index[i]);
    }
    for (size_t i = 0; i < network.num_branches(); ++i) {
        assert(network.edges_by_index[i]);
    }
    return true;
}

void assert_links_in_range(const Network& network) {
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        for (size_t j = 0; j < network.nodes_by_index[i]->links.size(); ++j) {
            assert(network.nodes_by_index[i]->links[j].edge_pmatrix_index < network.num_branches());
        }
    }
    for (size_t i = 0; i < network.num_branches(); ++i) {
        assert(network.edges_by_index[i]->link1->edge_pmatrix_index == i);
        assert(network.edges_by_index[i]->link2->edge_pmatrix_index == i);
    }
}

void assert_branch_lengths(AnnotatedNetwork& ann_network) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            assert(ann_network.fake_treeinfo->branch_lengths[p][i] >= ann_network.options.brlen_min);
        }
    }
}

NetworkState extract_network_state(AnnotatedNetwork &ann_network, bool extract_network) {
    assert_tip_links(ann_network.network);
    assert_links_in_range(ann_network.network);
    //assert_branch_lengths(ann_network);
    NetworkState state;
    state.brlen_linkage = ann_network.options.brlen_linkage;
    
    if (extract_network) {
        state.network = ann_network.network;
        state.network_valid = true;
    }
    assert_tip_links(state.network);
    
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        state.partition_brlens.resize(ann_network.fake_treeinfo->partition_count);
    } else {
        state.partition_brlens.resize(1);
    }
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        state.partition_brlen_scalers.resize(ann_network.fake_treeinfo->partition_count);
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            state.partition_brlen_scalers[p] = ann_network.fake_treeinfo->brlen_scalers[p];
        }
    }
    state.alphas.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        state.alphas[p] = ann_network.fake_treeinfo->alphas[p];
    }
    for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
        state.partition_brlens[p].resize(ann_network.network.num_branches());
        for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.num_branches(); ++pmatrix_index) {
            state.partition_brlens[p][pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    }

    state.partition_models.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t i = 0; i < state.partition_models.size(); ++i) {
        assign(state.partition_models[i], ann_network.fake_treeinfo->partitions[i]);
    }
    state.reticulation_probs = ann_network.reticulation_probs;
    if (extract_network) {
        assert(consecutive_indices(state.network));
        assert_tip_links(state.network);
    }
    return state;
}

void assert_rates(AnnotatedNetwork& ann_network) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        std::vector<double> recomputed_rates(ann_network.fake_treeinfo->partitions[p]->rate_cats);
        pll_compute_gamma_cats(ann_network.fake_treeinfo->alphas[p],
                ann_network.fake_treeinfo->partitions[p]->rate_cats, recomputed_rates.data(),
                ann_network.fake_treeinfo->gamma_mode[p]);
        for (size_t k = 0; k < recomputed_rates.size(); ++k) {
            assert(ann_network.fake_treeinfo->partitions[p]->rates[k] == recomputed_rates[k]);
        }
    }
}

void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state, bool copy_network) {
    ann_network.options.brlen_linkage = state.brlen_linkage;
    assert_tip_links(state.network);
    assert_links_in_range(state.network);
    if (copy_network) {
        assert(state.network_valid);
        ann_network.network = state.network;
    }
    for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
        for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.num_branches(); ++pmatrix_index) {
            ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] = state.partition_brlens[p][pmatrix_index];
            ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
    }
    for (size_t p = 0; p < state.partition_brlen_scalers.size(); ++p) {
        ann_network.fake_treeinfo->brlen_scalers[p] = state.partition_brlen_scalers[p];
    }
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        ann_network.fake_treeinfo->alphas[p] = state.alphas[p];
    }
    pllmod_treeinfo_update_prob_matrices(ann_network.fake_treeinfo, 1);

    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        for (size_t clv_index = 0; clv_index < ann_network.network.nodes.size(); ++clv_index) {
            ann_network.fake_treeinfo->clv_valid[p][clv_index] = 0;
        }
    }

    for (size_t i = 0; i < state.partition_models.size(); ++i) {
        assign(ann_network.fake_treeinfo->partitions[i], state.partition_models[i]);
    }

    ann_network.reticulation_probs = state.reticulation_probs;

    if (copy_network) {
        ann_network.travbuffer = reversed_topological_sort(ann_network.network);
        ann_network.blobInfo = partitionNetworkIntoBlobs(ann_network.network, ann_network.travbuffer);
        assert(consecutive_indices(state.network));
        assert(consecutive_indices(ann_network.network));
        assert_tip_links(ann_network.network);
        assert_links_in_range(ann_network.network);
        // invalidate all clv and pmatrix entries... TODO: can be optimized, only needs to be done if model or brlens changed
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
                ann_network.fake_treeinfo->clv_valid[p][i] = 0;
            }
            for (size_t i = 0; i < ann_network.network.edges.size(); ++i) {
                ann_network.fake_treeinfo->pmatrix_valid[p][i] = 0;
            }
        }
    }
    //assert_branch_lengths(ann_network);
    //assert_rates(ann_network);
}

bool reticulation_probs_equal(const NetworkState& old_state, const NetworkState& act_state) {
    assert(old_state.reticulation_probs.size() == act_state.reticulation_probs.size());
    bool all_fine = true;
    for (size_t j = 0; j < old_state.reticulation_probs.size(); ++j) {
        if (fabs(act_state.reticulation_probs[j] - old_state.reticulation_probs[j]) >= 1E-5) {
            std::cout << "wanted prob:\n";
            std::cout << "idx " << j << ": " << old_state.reticulation_probs[j] << "\n";
            std::cout << "\n";
            std::cout << "observed prob:\n";
            std::cout << "idx " << j << ": " << act_state.reticulation_probs[j] << "\n";
            std::cout << "\n";

            all_fine = false;
        }
    }
    return all_fine;
}

bool brlen_scalers_equal(const NetworkState& old_state, const NetworkState& act_state) {
    assert(old_state.partition_brlen_scalers.size() == act_state.partition_brlen_scalers.size());
    bool all_fine = true;
    for (size_t i = 0; i < act_state.partition_brlen_scalers.size(); ++i) {
        if (fabs(act_state.partition_brlen_scalers[i] - old_state.partition_brlen_scalers[i]) >= 1E-5) {
            return false;
        }
    }
    return true;
}

bool partition_brlens_equal(const NetworkState& old_state, const NetworkState& act_state) {
    assert(old_state.partition_brlens.size() == act_state.partition_brlens.size());
    bool all_fine = true;
    for (size_t i = 0; i < old_state.partition_brlens.size(); ++i) {
        assert(old_state.partition_brlens[i].size() == act_state.partition_brlens[i].size());
        for (size_t j = 0; j < act_state.partition_brlens[i].size(); ++j) {
            if (fabs(act_state.partition_brlens[i][j] - old_state.partition_brlens[i][j]) >= 1E-5) {
                std::cout << "wanted brlens:\n";
                for (size_t k = 0; k < old_state.partition_brlens[i].size(); ++k) {
                    std::cout << "idx " << k << ": " << old_state.partition_brlens[i][k] << "\n";
                }
                std::cout << "\n";
                std::cout << "observed brlens:\n";
                for (size_t k = 0; k < act_state.partition_brlens[i].size(); ++k) {
                    std::cout << "idx " << k << ": " << act_state.partition_brlens[i][k] << "\n";
                }
                std::cout << "\n";
                all_fine = false;
            }
        }
    }
    return all_fine;
}

bool topology_equal(const Network& n1, const Network& n2) {
    if (n1.num_branches() != n2.num_branches()) {
        return false;
    }
    for (size_t i = 0; i < n1.num_branches(); ++i) {
        if (n1.edges_by_index[i]->link1->node_clv_index != n2.edges_by_index[i]->link1->node_clv_index) {
            return false;
        }
        if (n1.edges_by_index[i]->link2->node_clv_index != n2.edges_by_index[i]->link2->node_clv_index) {
            return false;
        }
    }
    return true;
}

bool model_equal(const NetworkState& old_state, const NetworkState& act_state) {
    if (old_state.partition_models.size() != act_state.partition_models.size()) {
        return false;
    }
    for (size_t p = 0; p < old_state.partition_models.size(); ++p) {
        if (old_state.partition_models[p].to_string(true) != act_state.partition_models[p].to_string(true)) {
            return false;
        }
    }
    return true;
}

bool alphas_equal(const NetworkState& old_state, const NetworkState& act_state) {
    if (old_state.alphas.size() != act_state.alphas.size()) {
        return false;
    }
    for (size_t i = 0; i < old_state.alphas.size(); ++i) {
        if (old_state.alphas[i] == act_state.alphas[i]) {
            return false;
        }
    }
    return true;
}

bool network_states_equal(const NetworkState& old_state, const NetworkState& act_state) {
    return model_equal(old_state, act_state) && 
           ((!old_state.network_valid && !act_state.network_valid) || topology_equal(old_state.network, act_state.network)) && 
           reticulation_probs_equal(old_state, act_state) && 
           partition_brlens_equal(old_state, act_state) && 
           alphas_equal(old_state, act_state) &&
           brlen_scalers_equal(old_state, act_state);
}

AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options) {
    throw std::runtime_error("Not implemented yet");
}

}
