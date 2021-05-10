#include "PseudoLoglikelihood.hpp"

#include "../graph/NetworkTopology.hpp"
#include "Operation.hpp"

namespace netrax {

void merge_clvs(AnnotatedNetwork& ann_network, 
                size_t clv_index,
                const std::vector<double*>& tmp_clv_1, 
                const std::vector<double*>& tmp_clv_2, 
                const std::vector<double*>& tmp_clv_3,
                double weight_1,
                double weight_2,
                double weight_3,
                double weight_4
                )
{
    assert(weight_1 >= 0.0);
    assert(weight_1 <= 1.0);
    assert(weight_2 >= 0.0);
    assert(weight_2 <= 1.0);
    assert(weight_3 >= 0.0);
    assert(weight_3 <= 1.0);
    assert(weight_4 >= 0.0);
    assert(weight_4 <= 1.0);
    assert(weight_1 + weight_2 + weight_3 + weight_4 == 1.0);
    size_t fake_clv_index = ann_network.network.nodes.size();

    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        double* clv = ann_network.fake_treeinfo->partitions[p]->clv[clv_index];
        for (size_t i = 0; i < ann_network.partition_clv_ranges[p].inner_clv_num_entries; ++i) {
            double merged_entry = 0.0;

            if (weight_1 > 0.0) {
                merged_entry += weight_1 * tmp_clv_1[p][i];
            }

            if (weight_2 > 0.0) {
                merged_entry += weight_2 * tmp_clv_2[p][i];
            }

            if (weight_3 > 0.0) {
                merged_entry += weight_3 * tmp_clv_3[p][i];
            }

            if (weight_4 > 0.0) {
                merged_entry += weight_4 * ann_network.fake_treeinfo->partitions[p]->clv[fake_clv_index][i];
            }

            clv[i] = merged_entry;
        }
    }
}

double computePseudoLoglikelihood(AnnotatedNetwork& ann_network, int incremental, int update_pmatrices) {
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;

    fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    if (update_pmatrices) {
        pllmod_treeinfo_update_prob_matrices(ann_network.fake_treeinfo, !incremental);
    }

    // reuse the clv vectors in the treeinfo object for the final clvs

    size_t fake_clv_index = ann_network.network.nodes.size();
    size_t fake_pmatrix_index = ann_network.network.edges.size();

    for (size_t i = 0; i < ann_network.travbuffer.size(); ++i) {
        Node* node = ann_network.travbuffer[i];
        if (node->isTip()) {
            continue;
        }
        if (incremental && ann_network.pseudo_clv_valid[node->clv_index]) {
            continue;
        }

        // now we are computing the pseudo-clv...
        std::vector<Node*> children = getChildren(ann_network.network, node);
        assert(!children.empty());
        assert(children.size() <= 2);
        Node* left_child = children[0];
        Node* right_child = (children.size() == 1) ? nullptr : children[1];

        pll_operation_t take_both_op = buildOperation(ann_network.network, node, left_child, right_child, fake_clv_index, fake_pmatrix_index);
        pll_operation_t take_left_only_op = buildOperation(ann_network.network, node, left_child, nullptr, fake_clv_index, fake_pmatrix_index);
        pll_operation_t take_right_only_op = buildOperation(ann_network.network, node, nullptr, right_child, fake_clv_index, fake_pmatrix_index);

        double p_left = 1.0;
        double p_right = 1.0;
        if (left_child && left_child->getType() == NodeType::RETICULATION_NODE) {
            if (node == getReticulationFirstParent(ann_network.network, left_child)) {
                p_left = getReticulationFirstParentProb(ann_network, left_child);
            } else {
                assert(node == getReticulationSecondParent(ann_network.network, left_child));
                p_left = getReticulationSecondParentProb(ann_network, left_child);
            }
        }
        if (right_child && right_child->getType() == NodeType::RETICULATION_NODE) {
            if (node == getReticulationFirstParent(ann_network.network, right_child)) {
                p_right = getReticulationFirstParentProb(ann_network, right_child);
            } else {
                assert(node == getReticulationSecondParent(ann_network.network, right_child));
                p_right = getReticulationSecondParentProb(ann_network, right_child);
            }
        }
        double weight_1 = p_left * p_right;
        double weight_2 = p_left * (1.0 - p_right);
        double weight_3 = (1.0 - p_left) * p_right;
        double weight_4 = (1.0 - p_left) * (1.0 - p_right);

        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }

            pll_partition_t* partition = ann_network.fake_treeinfo->partitions[p];
            double* left_clv = (left_child) ? partition->clv[left_child->clv_index] : partition->clv[fake_clv_index];
            double* right_clv = (right_child) ? partition->clv[right_child->clv_index] : partition->clv[fake_clv_index];

            double* parent_clv_1 = ann_network.tmp_clv_1[p];
            double* parent_clv_2 = ann_network.tmp_clv_2[p];
            double* parent_clv_3 = ann_network.tmp_clv_3[p];

            assert(parent_clv_1);
            assert(parent_clv_2);
            assert(parent_clv_3);

            assert(ann_network.fake_treeinfo->partitions[p]->clv[node->clv_index]);
            assert(left_clv || left_child->isTip());
            assert(right_clv || right_child->isTip());

            unsigned int* parent_scaler = (node->scaler_index == -1) ? nullptr : partition->scale_buffer[node->scaler_index];
            unsigned int* left_scaler = (!left_child || left_child->scaler_index == -1) ? nullptr : partition->scale_buffer[left_child->scaler_index];
            unsigned int* right_scaler = (!right_child || right_child->scaler_index == -1) ? nullptr : partition->scale_buffer[right_child->scaler_index];

            if (weight_1 > 0.0) {
                // case 1: take both
                pll_update_partials_single(partition, &take_both_op, 1, parent_clv_1, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
            }
            if (weight_2 > 0.0) {
                // case 2: take left only
                pll_update_partials_single(partition, &take_left_only_op, 1, parent_clv_2, left_clv, partition->clv[fake_clv_index], parent_scaler, left_scaler, nullptr);
            }
            if (weight_3 > 0.0) {
                // case 3: take right only
                pll_update_partials_single(partition, &take_right_only_op, 1, parent_clv_3, partition->clv[fake_clv_index], right_clv, parent_scaler, nullptr, right_scaler);
            }
        }

        merge_clvs(ann_network, node->clv_index, ann_network.tmp_clv_1, ann_network.tmp_clv_2, ann_network.tmp_clv_3, weight_1, weight_2, weight_3, weight_4);

        ann_network.pseudo_clv_valid[node->clv_index] = true;
    }

    // Compute the pseudo loglikelihood at the root
    std::vector<double> partition_pseudo_logl(ann_network.fake_treeinfo->partition_count, 0.0);

    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        pll_partition_t* partition = ann_network.fake_treeinfo->partitions[p];
        // skip remote partitions
        if (!partition) {
            continue;
        }

        double* parent_clv = partition->clv[ann_network.network.root->clv_index];
        unsigned int* parent_scaler = (ann_network.network.root->scaler_index == -1) ? nullptr : partition->scale_buffer[ann_network.network.root->scaler_index];
        partition_pseudo_logl[p] = pll_compute_root_loglikelihood(partition, ann_network.network.root->clv_index, parent_clv, parent_scaler, ann_network.fake_treeinfo->param_indices[p], nullptr);
    }

    /* sum up likelihood from all threads */
    if (ann_network.fake_treeinfo->parallel_reduce_cb)
    {
        ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context,
                                    partition_pseudo_logl.data(),
                                    ann_network.fake_treeinfo->partition_count,
                                    PLLMOD_COMMON_REDUCE_SUM);
    }

    double pseudo_logl = std::accumulate(partition_pseudo_logl.begin(), partition_pseudo_logl.end(), 0.0);

    return pseudo_logl;
}

}