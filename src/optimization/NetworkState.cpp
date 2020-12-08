#include "NetworkState.hpp"
#include "../graph/NetworkTopology.hpp"

namespace netrax {

std::vector<std::vector<double> > extract_brlens_partitions(AnnotatedNetwork &ann_network) {
    std::vector<std::vector<double> > res;
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t n_partitions = 1;
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    res.resize(n_partitions);
    for (size_t p = 0; p < n_partitions; ++p) {
        res[p].resize(ann_network.network.edges.size());
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            assert(ann_network.network.edges_by_index[pmatrix_index] == &ann_network.network.edges[i]);
            res[p][pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    }
    return res;
}

std::vector<double> extract_brprobs(AnnotatedNetwork &ann_network) {
    std::vector<double> res;
    res.resize(ann_network.network.edges.size());
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
        res[pmatrix_index] = ann_network.network.edges[i].prob;
    }
    return res;
}


std::vector<double> extract_brlen_scalers(AnnotatedNetwork &ann_network) {
    std::vector<double> res;
    if (ann_network.options.brlen_linkage != PLLMOD_COMMON_BRLEN_SCALED) {
        return res;
    }
    res.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        res[p] = ann_network.fake_treeinfo->brlen_scalers[p];
    }
    return res;
}


NetworkState extract_network_state(AnnotatedNetwork &ann_network) {
    return NetworkState{extract_brlens_partitions(ann_network), extract_brprobs(ann_network), extract_brlen_scalers(ann_network)};
}


void apply_old_state(AnnotatedNetwork &ann_network,
        const std::vector<std::vector<double> > &old_brlens_partition, const std::vector<double>& old_brlen_scalers, const std::vector<double>& old_reticulation_probs) {
    std::vector<bool> visited(ann_network.network.nodes.size(), false);
    bool unlinkedMode = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    size_t n_partitions = 1;
    if (unlinkedMode) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    for (size_t p = 0; p < n_partitions; ++p) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            if (ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index]
                    != old_brlens_partition[p][pmatrix_index]) {
                ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] =
                        old_brlens_partition[p][pmatrix_index];
                ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
                invalidateHigherCLVs(ann_network,
                        getTarget(ann_network.network, &ann_network.network.edges[i]), true,
                        visited);
            }
        }
    }
    for (size_t p = 0; p < old_brlen_scalers.size(); ++p) {
        if (ann_network.fake_treeinfo->brlen_scalers[p] != old_brlen_scalers[p]) {
            ann_network.fake_treeinfo->brlen_scalers[p] = old_brlen_scalers[p];
            for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
                ann_network.fake_treeinfo->pmatrix_valid[p][i] = 0;
            }
            for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
                ann_network.fake_treeinfo->clv_valid[p][i] = 0;
            }
        }
    }
    for (size_t i = 0; i < old_reticulation_probs.size(); ++i) {
        if (ann_network.reticulation_probs[i] != old_reticulation_probs[i]) {
            ann_network.reticulation_probs[i] = old_reticulation_probs[i];
            invalidateHigherCLVs(ann_network, ann_network.network.root, true, visited);
        }
    }
}


void apply_network_state(AnnotatedNetwork &ann_network, const NetworkState &state) {
    apply_old_state(ann_network, state.brlens_partitions, state.brlen_scalers, state.reticulation_probs);
}


bool network_states_equal(NetworkState &act_network_state, NetworkState &old_network_state) {
    std::vector<std::vector<double> > act_brlens_partitions = act_network_state.brlens_partitions;
    std::vector<double> act_brprobs = act_network_state.reticulation_probs;
    std::vector<double> act_brlen_scalers = act_network_state.brlen_scalers;

    std::vector<std::vector<double> > old_brlens_partitions = old_network_state.brlens_partitions;
    std::vector<double> old_brprobs = old_network_state.reticulation_probs;
    std::vector<double> old_brlen_scalers = old_network_state.brlen_scalers;

    for (size_t i = 0; i < act_brlens_partitions.size(); ++i) {
        for (size_t j = 0; j < act_brlens_partitions[i].size(); ++j) {
            if (fabs(act_brlens_partitions[i][j] - old_brlens_partitions[i][j]) >= 1E-5) {
                std::cout << "wanted brlens:\n";
                for (size_t k = 0; k < old_brlens_partitions[i].size(); ++k) {
                    std::cout << "idx " << k << ": "
                            << old_brlens_partitions[i][k] << "\n";
                }
                std::cout << "\n";
                std::cout << "observed brlens:\n";
                for (size_t k = 0; k < act_brlens_partitions[i].size(); ++k) {
                    std::cout << "idx " << k << ": "
                            << act_brlens_partitions[i][k] << "\n";
                }
                std::cout << "\n";
            }
            assert(fabs(act_brlens_partitions[i][j] - old_brlens_partitions[i][j]) < 1E-5);
        }
    }

    for (size_t j = 0; j < act_brlen_scalers.size(); ++j) {
        if (fabs(act_brlen_scalers[j] - old_brlen_scalers[j]) >= 1E-5) {
            std::cout << "wanted brlen:\n";
            std::cout << "idx " << j << ": " << old_brlen_scalers[j] << "\n";
            std::cout << "\n";
            std::cout << "observed brlen:\n";
            std::cout << "idx " << j << ": " << act_brlen_scalers[j] << "\n";
            std::cout << "\n";
        }
        assert(fabs(act_brlen_scalers[j] - old_brlen_scalers[j]) < 1E-5);
    }
    
    for (size_t j = 0; j < act_brprobs.size(); ++j) {
        if (fabs(act_brprobs[j] - old_brprobs[j]) >= 1E-5) {
            std::cout << "wanted brprob:\n";
            std::cout << "idx " << j << ": " << old_brprobs[j] << "\n";
            std::cout << "\n";
            std::cout << "observed brprob:\n";
            std::cout << "idx " << j << ": " << act_brprobs[j] << "\n";
            std::cout << "\n";
        }
        assert(fabs(act_brprobs[j] - old_brprobs[j]) < 1E-5);
    }

    return true;
}

}
