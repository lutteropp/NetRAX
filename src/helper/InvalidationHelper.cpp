#include "Helper.hpp"

namespace netrax {

void invalidateSingleClv(AnnotatedNetwork& ann_network, unsigned int clv_index) {
    pllmod_treeinfo_t* treeinfo = ann_network.fake_treeinfo;
    for (size_t p = 0; p < treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!treeinfo->partitions[p]) {
            continue;
        }
        treeinfo->clv_valid[p][clv_index] = 0;
    }
    ann_network.pernode_displayed_tree_data[clv_index].num_active_displayed_trees = 0;
    if (ann_network.options.save_memory) {
        ann_network.pernode_displayed_tree_data[clv_index].displayed_trees.clear();
    }
    ann_network.pseudo_clv_valid[clv_index] = false;
    ann_network.cached_logl_valid = false;
}

void validateSingleClv(AnnotatedNetwork& ann_network, unsigned int clv_index) {
    pllmod_treeinfo_t* treeinfo = ann_network.fake_treeinfo;
    for (size_t p = 0; p < treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!treeinfo->partitions[p]) {
            continue;
        }
        treeinfo->clv_valid[p][clv_index] = 1;
    }
}

void invalidateHigherClvs(AnnotatedNetwork &ann_network, pllmod_treeinfo_t *treeinfo, const Node *node, bool invalidate_myself, std::vector<bool> &visited) {
    Network &network = ann_network.network;
    if (!node) {
        return;
    }
    if (!visited.empty() && visited[node->clv_index]) { // clv at node is already invalidated
        return;
    }
    if (invalidate_myself) {
        for (size_t p = 0; p < treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            treeinfo->clv_valid[p][node->clv_index] = 0;
        }
        ann_network.pernode_displayed_tree_data[node->clv_index].num_active_displayed_trees = 0;
        if (ann_network.options.save_memory) {
            ann_network.pernode_displayed_tree_data[node->clv_index].displayed_trees.clear();
        }
        ann_network.pseudo_clv_valid[node->clv_index] = false;
        if (!visited.empty()) {
            visited[node->clv_index] = true;
        }
    }
    if (node->clv_index == network.root->clv_index) {
        return;
    }
    if (node->type == NodeType::RETICULATION_NODE) {
        invalidateHigherClvs(ann_network, treeinfo, getReticulationFirstParent(network, node), true,
                visited);
        invalidateHigherClvs(ann_network, treeinfo, getReticulationSecondParent(network, node),
                true, visited);
    } else {
        invalidateHigherClvs(ann_network, treeinfo, getActiveParent(network, node), true, visited);
    }
    ann_network.cached_logl_valid = false;
}

void invalidateHigherPseudoClvs(AnnotatedNetwork &ann_network, pllmod_treeinfo_t *treeinfo, const Node *node, bool invalidate_myself, std::vector<bool> &visited) {
    Network &network = ann_network.network;
    if (!node) {
        return;
    }
    if (!visited.empty() && visited[node->clv_index]) { // clv at node is already invalidated
        return;
    }
    if (invalidate_myself) {
        ann_network.pseudo_clv_valid[node->clv_index] = false;
        if (!visited.empty()) {
            visited[node->clv_index] = true;
        }
    }
    if (node->clv_index == network.root->clv_index) {
        return;
    }
    if (node->type == NodeType::RETICULATION_NODE) {
        invalidateHigherClvs(ann_network, treeinfo, getReticulationFirstParent(network, node), true,
                visited);
        invalidateHigherClvs(ann_network, treeinfo, getReticulationSecondParent(network, node),
                true, visited);
    } else {
        invalidateHigherClvs(ann_network, treeinfo, getActiveParent(network, node), true, visited);
    }
    ann_network.cached_logl_valid = false;
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, const Node *node, bool invalidate_myself,
        std::vector<bool> &visited) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    invalidateHigherClvs(ann_network, treeinfo, node, invalidate_myself, visited);
}

void invalidateHigherPseudoCLVs(AnnotatedNetwork &ann_network, const Node *node, bool invalidate_myself,
        std::vector<bool> &visited) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    invalidateHigherPseudoClvs(ann_network, treeinfo, node, invalidate_myself, visited);
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, const Node *node, bool invalidate_myself) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    std::vector<bool> noVisited;
    invalidateHigherClvs(ann_network, treeinfo, node, invalidate_myself, noVisited);
}

void invalidateHigherPseudoCLVs(AnnotatedNetwork &ann_network, Node *node, bool invalidate_myself) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    std::vector<bool> noVisited;
    invalidateHigherPseudoClvs(ann_network, treeinfo, node, invalidate_myself, noVisited);
}

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index,
        std::vector<bool> &visited) {
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    for (size_t p = 0; p < treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!treeinfo->partitions[p]) {
            continue;
        }
        treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
    }
    invalidateHigherCLVs(ann_network,
            getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]), true,
            visited);
}

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index) {
    std::vector<bool> noVisited;
    invalidatePmatrixIndex(ann_network, pmatrix_index, noVisited);
}

void invalidPmatrixIndexOnly(AnnotatedNetwork& ann_network, size_t pmatrix_index) {
    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
            continue;
        }
        ann_network.fake_treeinfo->pmatrix_valid[partition_idx][pmatrix_index] = 0;
    }
    ann_network.cached_logl_valid = false;
}

bool allClvsValid(pllmod_treeinfo_t* treeinfo, size_t clv_index) {
    for (size_t p = 0; p < treeinfo->partition_count; ++p) {
        if (treeinfo->partitions[p]) {
            if (!treeinfo->clv_valid[p][clv_index]) {
                return false;
            }
        }
    }
    return true;
}

void invalidate_pmatrices(AnnotatedNetwork &ann_network,
        std::vector<size_t> &affectedPmatrixIndices) {
    pllmod_treeinfo_t *fake_treeinfo = ann_network.fake_treeinfo;
    for (size_t pmatrix_index : affectedPmatrixIndices) {
        assert(ann_network.network.edges_by_index[pmatrix_index]);
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip reote partitions
            if (!fake_treeinfo->partitions[p]) {
                continue;
            }
            fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
    }
    pllmod_treeinfo_update_prob_matrices(fake_treeinfo, 0);
}

void invalidateAllCLVs(AnnotatedNetwork &ann_network) {
    for (size_t i = ann_network.network.num_tips(); i < ann_network.network.num_nodes(); ++i) {
       invalidateSingleClv(ann_network, i);
    }
}

}