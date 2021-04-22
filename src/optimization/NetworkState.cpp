#include <algorithm>

#include "NetworkState.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../likelihood/DisplayedTreeData.hpp"

namespace netrax {

bool assert_tip_links(const Network& network) {
    for (size_t i = 0; i < network.num_tips(); ++i) {
        assert(network.nodes_by_index[i]->clv_index == i);
        assert(!network.nodes_by_index[i]->label.empty());
        assert(network.nodes_by_index[i]->links.size() == 1);
    }
    return true;
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

bool assert_links_in_range(const Network& network) {
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        for (size_t j = 0; j < network.nodes_by_index[i]->links.size(); ++j) {
            assert(network.nodes_by_index[i]->links[j].edge_pmatrix_index < network.num_branches());
        }
    }
    for (size_t i = 0; i < network.num_branches(); ++i) {
        assert(network.edges_by_index[i]->link1->edge_pmatrix_index == i);
        assert(network.edges_by_index[i]->link2->edge_pmatrix_index == i);
    }
    return true;
}

bool assert_branch_lengths(AnnotatedNetwork& ann_network) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            assert(ann_network.fake_treeinfo->branch_lengths[p][i] >= ann_network.options.brlen_min);
        }
    }
    return true;
}

void apply_displayed_trees_data(const NetworkState& state, AnnotatedNetwork& ann_network, bool copy_network) {
    for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
        ann_network.pernode_displayed_tree_data[i] = state.pernode_displayed_tree_data[i];
    }
}


void add_missing_displayed_trees_data(const AnnotatedNetwork& ann_network, NetworkState& state) {
    size_t maxReticulations = ann_network.options.max_reticulations;
    for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
        while (state.pernode_displayed_tree_data[i].displayed_trees.size() < ann_network.pernode_displayed_tree_data[i].num_active_displayed_trees) {
            // add another displayed tree struct
            state.pernode_displayed_tree_data[i].displayed_trees.emplace_back(DisplayedTreeData(ann_network.fake_treeinfo, ann_network.partition_clv_ranges, ann_network.partition_scale_buffer_ranges, maxReticulations));
        }
    }
}

void extract_displayed_trees_data(const AnnotatedNetwork& ann_network, NetworkState& state) {
    add_missing_displayed_trees_data(ann_network, state);
    for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
        state.pernode_displayed_tree_data[i] = ann_network.pernode_displayed_tree_data[i];
    }
}

bool neighborsSame(const Network& n1, const Network& n2) {
    bool same = true;
    for (size_t i = 0; i < n1.num_nodes(); ++i) {
        std::vector<Node*> n1_neighbors = getNeighbors(n1, n1.nodes_by_index[i]);
        std::vector<size_t> n1_neigh_indices(n1_neighbors.size());
        for (size_t j = 0; j < n1_neighbors.size(); ++j) {
            n1_neigh_indices[j] = n1_neighbors[j]->clv_index;
        }
        std::sort(n1_neigh_indices.begin(), n1_neigh_indices.end());

        std::vector<Node*> n2_neighbors = getNeighbors(n2, n2.nodes_by_index[i]);
        std::vector<size_t> n2_neigh_indices(n2_neighbors.size());
        for (size_t j = 0; j < n2_neighbors.size(); ++j) {
            n2_neigh_indices[j] = n2_neighbors[j]->clv_index;
        }
        std::sort(n2_neigh_indices.begin(), n2_neigh_indices.end());

        same &= (n1_neigh_indices == n2_neigh_indices);
    }
    return same;
}

void extract_network_state(AnnotatedNetwork &ann_network, NetworkState& state_to_reuse, bool extract_network) {
    assert(assert_tip_links(ann_network.network));
    assert(assert_links_in_range(ann_network.network));
    //assert_branch_lengths(ann_network);
    state_to_reuse.brlen_linkage = ann_network.options.brlen_linkage;
    
    if (extract_network) {
        assert(ann_network.network.root);
        state_to_reuse.network = ann_network.network;
        assert(state_to_reuse.network.root);
        state_to_reuse.network_valid = true;
    } else {
        state_to_reuse.network_valid = false;
    }
    assert(assert_tip_links(state_to_reuse.network));
    
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            state_to_reuse.partition_brlen_scalers[p] = ann_network.fake_treeinfo->brlen_scalers[p];
        }
    }
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        state_to_reuse.alphas[p] = ann_network.fake_treeinfo->alphas[p];
    }
    for (size_t p = 0; p < state_to_reuse.partition_brlens.size(); ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.num_branches(); ++pmatrix_index) {
            state_to_reuse.partition_brlens[p][pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    }

    for (size_t i = 0; i < state_to_reuse.partition_models.size(); ++i) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[i]) {
            continue;
        }
        assign(state_to_reuse.partition_models[i], ann_network.fake_treeinfo->partitions[i]);
    }
    state_to_reuse.reticulation_probs = ann_network.reticulation_probs;
    if (extract_network) {
        assert(consecutive_indices(state_to_reuse.network));
        assert(assert_tip_links(state_to_reuse.network));
    }
    extract_displayed_trees_data(ann_network, state_to_reuse);
    state_to_reuse.n_trees = (1 << ann_network.network.num_reticulations());
    state_to_reuse.n_branches = ann_network.network.num_branches();

    state_to_reuse.cached_logl = ann_network.cached_logl;
    state_to_reuse.cached_logl_valid = ann_network.cached_logl_valid;

    if (extract_network) {
        assert(neighborsSame(ann_network.network, state_to_reuse.network));
    }
}

NetworkState extract_network_state(AnnotatedNetwork &ann_network, bool extract_network) {
    NetworkState state;
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        state.partition_brlens.resize(ann_network.fake_treeinfo->partition_count);
    } else {
        state.partition_brlens.resize(1);
    }
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        state.partition_brlen_scalers.resize(ann_network.fake_treeinfo->partition_count);
    }
    state.alphas.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        state.partition_brlens[p].resize(ann_network.network.edges.size());
    }
    state.partition_models.resize(ann_network.fake_treeinfo->partition_count);
    
    state.displayed_tree_clv_ranges.resize(ann_network.fake_treeinfo->partition_count);
    state.displayed_tree_scale_buffer_ranges.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        state.displayed_tree_clv_ranges[p] = get_clv_range(ann_network.fake_treeinfo->partitions[p]);
        state.displayed_tree_scale_buffer_ranges[p] = get_scale_buffer_range(ann_network.fake_treeinfo->partitions[p]);
    }
    state.pernode_displayed_tree_data.resize(ann_network.network.nodes.size());
    
    for (size_t i = 0; i < ann_network.network.num_tips(); ++i) {
        std::vector<double*> tip_clv(ann_network.fake_treeinfo->partition_count, nullptr);
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            tip_clv[p] = ann_network.fake_treeinfo->partitions[p]->clv[i];
        }
        state.pernode_displayed_tree_data[i].displayed_trees.emplace_back(DisplayedTreeData(ann_network.fake_treeinfo, ann_network.partition_clv_ranges, ann_network.partition_scale_buffer_ranges, tip_clv, ann_network.options.max_reticulations));
        state.pernode_displayed_tree_data[i].num_active_displayed_trees = 1;
    }

    extract_network_state(ann_network, state, extract_network);
    return state;
}

bool assert_rates(AnnotatedNetwork& ann_network) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        std::vector<double> recomputed_rates(ann_network.fake_treeinfo->partitions[p]->rate_cats);
        pll_compute_gamma_cats(ann_network.fake_treeinfo->alphas[p],
                ann_network.fake_treeinfo->partitions[p]->rate_cats, recomputed_rates.data(),
                ann_network.fake_treeinfo->gamma_mode[p]);
        for (size_t k = 0; k < recomputed_rates.size(); ++k) {
            assert(ann_network.fake_treeinfo->partitions[p]->rates[k] == recomputed_rates[k]);
        }
    }
    return true;
}

void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state, bool copy_network) {
    //ann_network.options.brlen_linkage = state.brlen_linkage;
    assert(assert_tip_links(state.network));
    assert(assert_links_in_range(state.network));
    if (copy_network) {
        assert(state.network_valid);
        assert(state.network.root);
        ann_network.network = state.network;
        ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    }
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.num_branches(); ++pmatrix_index) {
                if (ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] != state.partition_brlens[p][pmatrix_index]) {
                    ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] = state.partition_brlens[p][pmatrix_index];
                    ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
                }
            }
        }
    } else {
        for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.num_branches(); ++pmatrix_index) {
                if (ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] != state.partition_brlens[0][pmatrix_index]) {
                    ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] = state.partition_brlens[0][pmatrix_index];
                    ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
                }
            }
        }
    }
    for (size_t p = 0; p < state.partition_brlen_scalers.size(); ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        ann_network.fake_treeinfo->brlen_scalers[p] = state.partition_brlen_scalers[p];
    }
    for (size_t p = 0; p < state.alphas.size(); ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        ann_network.fake_treeinfo->alphas[p] = state.alphas[p];
    }
    for (size_t i = 0; i < state.partition_models.size(); ++i) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[i]) {
            continue;
        }
        assign(ann_network.fake_treeinfo->partitions[i], state.partition_models[i]);
    }
    ann_network.reticulation_probs = state.reticulation_probs;
    pllmod_treeinfo_update_prob_matrices(ann_network.fake_treeinfo, 1); // this (full pmatrix recomputation) is needed if the model parameters changed

    if (copy_network) {
        assert(consecutive_indices(state.network));
        assert(consecutive_indices(ann_network.network));
        assert(assert_tip_links(ann_network.network));
        assert(assert_links_in_range(ann_network.network));
    }
    apply_displayed_trees_data(state, ann_network, copy_network);
    // set all clvs to valid
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        for (size_t clv_index = 0; clv_index < ann_network.network.nodes.size(); ++clv_index) {
            ann_network.fake_treeinfo->clv_valid[p][clv_index] = 1;
        }
    }
    ann_network.cached_logl = state.cached_logl;
    ann_network.cached_logl_valid = state.cached_logl_valid;

    assert(assert_branch_lengths(ann_network));
    assert(assert_rates(ann_network));

    if (copy_network) {
        assert(neighborsSame(ann_network.network, state.network));
    }
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

            std::cout << "reticulation probs not equal\n";
            all_fine = false;
            break;
        }
    }
    return all_fine;
}

bool brlen_scalers_equal(const NetworkState& old_state, const NetworkState& act_state) {
    assert(old_state.partition_brlen_scalers.size() == act_state.partition_brlen_scalers.size());
    for (size_t i = 0; i < act_state.partition_brlen_scalers.size(); ++i) {
        if (fabs(act_state.partition_brlen_scalers[i] - old_state.partition_brlen_scalers[i]) >= 1E-5) {
            std::cout << "brlen scalers not equal\n";
            return false;
        }
    }
    return true;
}

bool partition_brlens_equal(const NetworkState& old_state, const NetworkState& act_state) {
    assert(old_state.partition_brlens.size() == act_state.partition_brlens.size());
    assert(old_state.n_branches == act_state.n_branches);
    bool all_fine = true;
    for (size_t i = 0; i < old_state.partition_brlens.size(); ++i) {
        for (size_t j = 0; j < act_state.n_branches; ++j) {
            if (fabs(act_state.partition_brlens[i][j] - old_state.partition_brlens[i][j]) >= 1E-5) {
                std::cout << "wanted brlens:\n";
                for (size_t k = 0; k < old_state.n_branches; ++k) {
                    std::cout << "idx " << k << ": " << old_state.partition_brlens[i][k] << "\n";
                }
                std::cout << "\n";
                std::cout << "observed brlens:\n";
                for (size_t k = 0; k < act_state.n_branches; ++k) {
                    std::cout << "idx " << k << ": " << act_state.partition_brlens[i][k] << "\n";
                }
                std::cout << "\n";
                std::cout << "brlens not equal\n";
                all_fine = false;
            }
        }
    }
    return all_fine;
}

bool topology_equal(const Network& n1, const Network& n2) {
    if (n1.num_branches() != n2.num_branches()) {
        std::cout << "topology not equal: different num branches \n";
        return false;
    }
    for (size_t i = 0; i < n1.num_branches(); ++i) {
        if (n1.edges_by_index[i]->link1->node_clv_index != n2.edges_by_index[i]->link1->node_clv_index) {
            std::cout << "topology not equal\n";
            return false;
        }
        if (n1.edges_by_index[i]->link2->node_clv_index != n2.edges_by_index[i]->link2->node_clv_index) {
            std::cout << "topology not equal\n";
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
            std::cout << "model not equal\n";
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
        if (old_state.alphas[i] != act_state.alphas[i]) {
            std::cout << "alphas not equal\n";
            return false;
        }
    }
    return true;
}

bool displayed_trees_equal(const NetworkState& old_state, const NetworkState& act_state) {
    assert(old_state.pernode_displayed_tree_data.size() == act_state.pernode_displayed_tree_data.size());
    for (size_t i = 0; i < old_state.pernode_displayed_tree_data.size(); ++i) {
        if (old_state.pernode_displayed_tree_data[i] != act_state.pernode_displayed_tree_data[i]) {
            return false;
        }
    }
    return true;
}

bool network_states_equal(const NetworkState& old_state, const NetworkState& act_state) {
    if (old_state.network_valid != act_state.network_valid) {
        std::cout << "different network valid\n";
        if (old_state.network_valid) {
            std::cout << "old state has valid network, act state doesn't\n";
        } else {
            std::cout << "act state has valid network, old state doesn't\n";
        }
    }
    return model_equal(old_state, act_state) && 
           ((!old_state.network_valid && !act_state.network_valid) || topology_equal(old_state.network, act_state.network)) && 
           reticulation_probs_equal(old_state, act_state) && 
           partition_brlens_equal(old_state, act_state) && 
           alphas_equal(old_state, act_state) &&
           brlen_scalers_equal(old_state, act_state) &&
           displayed_trees_equal(old_state, act_state);
}

AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options) {
    throw std::runtime_error("Not implemented yet");
}

}
