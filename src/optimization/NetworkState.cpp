#include "NetworkState.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../likelihood/LikelihoodComputation.hpp"

namespace netrax {

NetworkState extract_network_state(AnnotatedNetwork &ann_network) {
    NetworkState state;
    
    state.network = cloneNetwork(ann_network.network);
    
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        state.partition_brlens.resize(ann_network.fake_treeinfo->partition_count);
    } else {
        state.partition_brlens.resize(1);
    }
    for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
        state.partition_brlens[p].resize(ann_network.network.edges.size()+1);
        for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.edges.size(); ++pmatrix_index) {
            state.partition_brlens[p][pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    }

    state.partition_models.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t i = 0; i < state.partition_models.size(); ++i) {
        assign(state.partition_models[i], ann_network.fake_treeinfo->partitions[i]);
    }
    state.reticulation_probs = ann_network.reticulation_probs;
    return state;
}

void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state) {
    ann_network.network = cloneNetwork(state.network);
    for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
        for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.edges.size(); ++pmatrix_index) {
            ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] = state.partition_brlens[p][pmatrix_index];
            ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
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

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(ann_network.network, ann_network.travbuffer);
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

bool partition_brlens_equal(const NetworkState& old_state, const NetworkState& act_state) {
    assert(old_state.partition_brlens.size() == act_state.partition_brlens.size());
    bool all_fine = true;
    for (size_t i = 0; i < act_state.partition_brlens.size(); ++i) {
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

bool network_states_equal(const NetworkState& old_state, const NetworkState &act_state) {
    // TODO: Also check for model equality
    return reticulation_probs_equal(old_state, act_state) && partition_brlens_equal(old_state, act_state);
}

AnnotatedNetwork build_annotated_network_from_state(NetworkState& state, const NetraxOptions& options) {
    throw std::runtime_error("Not implemented yet");
}

}