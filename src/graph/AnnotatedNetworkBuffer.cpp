#include "AnnotatedNetworkBuffer.hpp"

/*
namespace netrax {

std::vector<std::vector<double> > extract_brlens(const AnnotatedNetwork &ann_network) {
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
            if (n_partitions == 1) {
                assert(ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] == ann_network.network.edges_by_index[pmatrix_index]->length);
            }
            res[p][pmatrix_index] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    }
    return res;
}

std::vector<std::vector<double> > extract_brprobs(const AnnotatedNetwork &ann_network) {
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
            res[p][pmatrix_index] = ann_network.network.edges[i].prob;
        }
    }
    return res;
}

std::vector<std::vector<double> > extract_brlen_scalers(const AnnotatedNetwork &ann_network) {
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
            res[p][pmatrix_index] = ann_network.fake_treeinfo->brlen_scalers[p][pmatrix_index];
        }
    }
    return res;
}

std::vector<std::vector<std::vector<double> > > extract_pmatrix(const AnnotatedNetwork &ann_network) {
    std::vector<std::vector<double> > res;
    ann_network.fake_treeinfo->partitions[0]->pmatrix;
}

std::vector<std::vector<double> > extract_clvs(const AnnotatedNetwork &ann_network) {

}

void apply_pmatrix(AnnotatedNetwork &ann_network, std::vector<std::vector<double> >& pmatrix) {

}

void apply_clvs(AnnotatedNetwork &ann_network, std::vector<std::vector<double> >& clvs) {

}

void apply_brlens(AnnotatedNetwork &ann_network,
        const std::vector<std::vector<double> > &old_brlens) {
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
                    != old_brlens[p][pmatrix_index]) {
                ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] =
                        old_brlens[p][pmatrix_index];
                if (n_partitions == 1) {
                    ann_network.network.edges_by_index[pmatrix_index]->length = old_brlens[p][pmatrix_index];
                }
                ann_network.fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
                invalidateHigherCLVs(ann_network,
                        getTarget(ann_network.network, &ann_network.network.edges[i]), true,
                        visited);
            }
        }
    }
}

struct AnnotatedNetworkState {
    std::vector<std::vector<double> > brlens;
    std::vector<std::vector<double> > brprobs;
    std::vector<std::vector<double> > brlen_scalers;
    std::vector<std::vector<double> > pmatrix;
    std::vector<std::vector<double> > clv;
    BlobInformation blobInfo;
    std::vector<Node*> travbuffer;
};

AnnotatedNetworkState extractState(const AnnotatedNetwork& ann_network) {
    AnnotatedNetworkState state;
    state.brlens = extract_brlens(ann_network);
    state.brprobs = extract_brprobs(ann_network);
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        state.brlen_scalers = extract_brlen_scalers(ann_network);
    }
    state.pmatrix = extract_pmatrix(ann_network);
    state.clv = extract_clv(ann_network);
    state.blobInfo = extract_blobinfo(ann_network);
    state.travbuffer = extract_travbuffer(ann_network);
    return state;
}

void applyState(AnnotatedNetwork& ann_network, AnnotatedNetworkState& state);
    apply_brlens(ann_network, state.brlens);
    apply_brprobs(ann_network, state.brprobs);
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        apply_brlen_scalers(ann_network, state.brlen_scalers);
    }
    apply_pmatrix(ann_network, state.pmatrix);
    apply_clv(ann_network, state.clv);
    apply_blobinfo(ann_network, state.blobInfo);
    apply_travbuffer(ann_network, state.travbuffer);
}
*/