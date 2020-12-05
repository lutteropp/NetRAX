/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../graph/BiconnectedComponents.hpp"
#include "../graph/Node.hpp"
#include "../DebugPrintFunctions.hpp"
#include "Operations.hpp"
#include "DisplayedTreeData.hpp"

#include <cassert>
#include <cmath>
#include <mpreal.h>

namespace netrax {

size_t findReticulationIndexInNetwork(Network &network, Node *retNode) {
    assert(retNode);
    assert(retNode->type == NodeType::RETICULATION_NODE);
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        if (network.reticulation_nodes[i]->clv_index == retNode->clv_index) {
            return i;
        }
    }
    throw std::runtime_error("Reticulation not found in network");
}

mpfr::mpreal displayed_tree_prob(AnnotatedNetwork &ann_network, size_t megablob_idx,
        size_t partition_index) {
    if (ann_network.fake_treeinfo->brlen_linkage != PLLMOD_COMMON_BRLEN_UNLINKED) {
        partition_index = 0;
    }
    BlobInformation &blobInfo = ann_network.blobInfo;
    mpfr::mpreal logProb = 0;
    for (size_t i = 0; i < blobInfo.reticulation_nodes_per_megablob[megablob_idx].size(); ++i) {
        size_t active_pmatrix_idx = getReticulationActiveParentPmatrixIndex(
                blobInfo.reticulation_nodes_per_megablob[megablob_idx][i]);
        mpfr::mpreal prob = ann_network.branch_probs[partition_index][active_pmatrix_idx];
        logProb += mpfr::log(prob);
    }
    return mpfr::exp(logProb);
}

std::vector<bool> init_clv_touched(AnnotatedNetwork& ann_network, bool incremental, int partition_idx) {
    std::vector<bool> clv_touched(ann_network.network.nodes.size() + 1, false);
    for (size_t i = 0; i < ann_network.network.num_tips(); ++i) {
        clv_touched[i] = true;
    }
    if (ann_network.network.num_reticulations() == 0 && incremental) {
        for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
            if (ann_network.fake_treeinfo->clv_valid[partition_idx][i]) {
                clv_touched[i] = true;
            }
        }
    }
    for (size_t i = 0; i < ann_network.blobInfo.megablob_roots.size(); ++i) {
        if (ann_network.fake_treeinfo->clv_valid[partition_idx][ann_network.blobInfo.megablob_roots[i]->clv_index]) {
            clv_touched[ann_network.blobInfo.megablob_roots[i]->clv_index] = true;
        }
    }
    clv_touched[ann_network.network.nodes.size()] = true; // fake clv index
    return clv_touched;
}


void setup_pmatrices(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    /* NOTE: in unlinked brlen mode, up-to-date brlens for partition p
     * have to be prefetched to treeinfo->branch_lengths[p] !!! */
    bool collect_brlen =
            (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? false : true);
    if (collect_brlen) {
        for (size_t i = 0; i < network.edges.size() + 1; ++i) { // +1 for the the fake entry
            fake_treeinfo.branch_lengths[0][i] = 0.0;
            ann_network.branch_probs[0][i] = 1.0;
        }
        for (size_t i = 0; i < network.num_branches(); ++i) {
            fake_treeinfo.branch_lengths[0][network.edges[i].pmatrix_index] =
                    network.edges[i].length;
            ann_network.branch_probs[0][network.edges[i].pmatrix_index] = network.edges[i].prob;
        }
        if (update_pmatrices) {
            pllmod_treeinfo_update_prob_matrices(&fake_treeinfo, !incremental);
        }
    }
}


double displayed_tree_logprob(AnnotatedNetwork &ann_network, size_t tree_index,
        size_t partition_index) {
    if (ann_network.options.brlen_linkage != PLLMOD_COMMON_BRLEN_UNLINKED) {
        partition_index = 0;
    }
    Network &network = ann_network.network;
    setReticulationParents(network, tree_index);
    mpfr::mpreal logProb = 0;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        size_t active_pmatrix_idx = getReticulationActiveParentPmatrixIndex(
                network.reticulation_nodes[i]);
        mpfr::mpreal prob = ann_network.branch_probs[partition_index][active_pmatrix_idx];
        logProb += mpfr::log(prob);
    }
    return logProb.toDouble();
}

DisplayedTreeData compute_displayed_tree(AnnotatedNetwork &ann_network, std::vector<bool> &clv_touched,
        std::vector<bool> &dead_nodes, Node *displayed_tree_root, bool incremental,
        const std::vector<Node*> &parent, pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx,
        size_t partition_idx, Node *startNode = nullptr) {
    Network &network = ann_network.network;
    double tree_logl = 0.0;
    double tree_logprob = displayed_tree_logprob(ann_network, tree_idx, partition_idx);
    std::vector<double> tree_persite_logl(fake_treeinfo.partitions[partition_idx]->sites, 0.0);
    unsigned int states_padded = fake_treeinfo.partitions[partition_idx]->states_padded;
    unsigned int sites = fake_treeinfo.partitions[partition_idx]->sites;
    unsigned int rate_cats = fake_treeinfo.partitions[partition_idx]->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;
    std::vector<double> tree_clv(clv_len, 0.0);

    std::vector<pll_operation_t> ops;
    if (startNode) {
        ops = createOperationsUpdatedReticulation(ann_network, partition_idx, parent, startNode,
                dead_nodes, incremental, displayed_tree_root);
    } else {
        ops = createOperations(ann_network, partition_idx, parent, ann_network.blobInfo, 0,
                dead_nodes, incremental, displayed_tree_root);
    }
    unsigned int ops_count = ops.size();
    if (ops_count == 0) {
        std::cout << "No operations found\n";
        assert(false);
        assert(tree_logl < 0);
        return DisplayedTreeData{tree_idx, tree_logl, tree_logprob, tree_clv, tree_persite_logl};
    }
    std::vector<bool> will_be_touched = clv_touched;
    for (size_t i = 0; i < ops_count; ++i) {
        will_be_touched[ops[i].parent_clv_index] = true;
        if (!will_be_touched[ops[i].child1_clv_index]) {
            std::cout << "problematic clv index: " << ops[i].child1_clv_index << "\n";
            std::cout << exportDebugInfo(network) << "\n";
        }
        assert(will_be_touched[ops[i].child1_clv_index]);
        if (!will_be_touched[ops[i].child2_clv_index]) {
            std::cout << "problematic clv index: " << ops[i].child2_clv_index << "\n";
        }
        assert(will_be_touched[ops[i].child2_clv_index]);
    }

    Node *ops_root = network.nodes_by_index[ops[ops.size() - 1].parent_clv_index];

    pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);
    for (size_t i = 0; i < ops_count; ++i) {
        clv_touched[ops[i].parent_clv_index] = true;
    }

    bool toplevel_trifurcation = (getChildren(network, network.root).size() == 3);
    assert(!toplevel_trifurcation);
    if (toplevel_trifurcation) {
        Node *rootBack = getTargetNode(network, ops_root->getLink());
        if (ops_root == network.root && !dead_nodes[rootBack->clv_index]) {
            tree_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx],
                    ops_root->clv_index, ops_root->scaler_index, rootBack->clv_index,
                    rootBack->scaler_index, ops_root->getLink()->edge_pmatrix_index,
                    fake_treeinfo.param_indices[partition_idx],
                    tree_persite_logl.empty() ? nullptr : tree_persite_logl.data());
        } else {
            tree_logl = pll_compute_root_loglikelihood(fake_treeinfo.partitions[partition_idx],
                    ops_root->clv_index, ops_root->scaler_index,
                    fake_treeinfo.param_indices[partition_idx],
                    tree_persite_logl.empty() ? nullptr : tree_persite_logl.data());
        }
    } else {
        tree_logl = pll_compute_root_loglikelihood(fake_treeinfo.partitions[partition_idx],
                ops_root->clv_index, ops_root->scaler_index,
                fake_treeinfo.param_indices[partition_idx],
                tree_persite_logl.empty() ? nullptr : tree_persite_logl.data());
    }

    assert(tree_logl < 0);
    return DisplayedTreeData{tree_idx, tree_logl, tree_logprob, tree_clv, tree_persite_logl};
}

std::vector<DisplayedTreeData> process_partition_new(AnnotatedNetwork &ann_network, int partition_idx, int incremental, 
        const std::vector<Node*> &parent, std::vector<bool> *touched) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);

    std::vector<bool> clv_touched = init_clv_touched(ann_network, incremental, partition_idx);
    size_t n_trees = 1 << network.num_reticulations();
    std::vector<DisplayedTreeData> displayed_trees;
    displayed_trees.reserve(n_trees);

    DisplayedTreeData last_tree;

    setReticulationParents(network, 0);
    for (size_t i = 0; i < n_trees; ++i) {
        size_t tree_idx = i;
        Node *start_node = nullptr;

        tree_idx = i ^ (i >> 1);
        if (i > 0) {
            size_t lastI = i - 1;
            size_t last_tree_idx = lastI ^ (lastI >> 1);
            size_t only_changed_bit = tree_idx ^ last_tree_idx;
            size_t changed_bit_pos = log2(only_changed_bit);
            bool changed_bit_is_set = tree_idx & only_changed_bit;
            start_node = network.reticulation_nodes[changed_bit_pos];
            start_node->getReticulationData()->setActiveParentToggle(changed_bit_is_set);
        }
        // TODO: Don't we need to invalidate the higher clvs here?

        Node *displayed_tree_root = nullptr;
        std::vector<bool> dead_nodes = collect_dead_nodes(network, network.root->clv_index, &displayed_tree_root);

        if (start_node && dead_nodes[start_node->clv_index]) {
            //std::cout << "The start node is a dead node!!!\n";
            displayed_trees.emplace_back(last_tree);
            continue;
            // TODO: what to do here? Is it correct just to return the tree from before?
        }

        if (start_node && !clv_touched[start_node->clv_index]) {
            fill_untouched_clvs(ann_network, clv_touched, dead_nodes, partition_idx,
                    start_node);
            assert(clv_touched[start_node->clv_index]);
        }
        DisplayedTreeData act_tree = compute_displayed_tree(ann_network, clv_touched, dead_nodes, displayed_tree_root,
                incremental, parent, fake_treeinfo, tree_idx, partition_idx, start_node);

        displayed_trees.emplace_back(act_tree);
        last_tree = act_tree;
    }

    return displayed_trees;
}

bool update_reticulation_probs_unlinked(AnnotatedNetwork &ann_network, const std::vector<DisplayedTreeData>& displayed_trees, int partition_idx) {
    bool changed = false;
    size_t n_sites = displayed_trees[0].tree_persite_logl.size();
    unsigned int n_sites_total = ann_network.fake_treeinfo->partitions[partition_idx]->pattern_weight_sum;
    std::vector<double> best_persite_logl(n_sites, -std::numeric_limits<double>::infinity());
    std::vector<size_t> best_tree_idx(n_sites, 0);
    for (const auto &tree : displayed_trees) {
        for (size_t i = 0; i < tree.tree_persite_logl.size(); ++i) {
            if (tree.tree_persite_logl[i] > best_persite_logl[i]) {
                best_persite_logl[i] = tree.tree_persite_logl[i];
                best_tree_idx[i] = tree.tree_idx;
            }
        }
    }
    std::vector<size_t> total_taken(ann_network.network.num_reticulations(), 0);
    for (size_t i = 0; i < n_sites; ++i) {
        unsigned int site_weight = ann_network.fake_treeinfo->partitions[partition_idx]->pattern_weights[i];
        // find the reticulation nodes that have taken their first parent in the best tree
        for (size_t j = 0; j < ann_network.network.num_reticulations(); ++j) {
            if (!(best_tree_idx[i] & (1 << j))) {
                total_taken[j] += site_weight;
            }
        }
    }

    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        size_t pmatrix_index = getReticulationFirstParentPmatrixIndex(ann_network.network.reticulation_nodes[i]);
        double old_prob = ann_network.branch_probs[partition_idx][pmatrix_index];
        double new_prob = (double) total_taken[i] / n_sites_total;
        ann_network.branch_probs[partition_idx][pmatrix_index] = new_prob;
        changed |= (old_prob != new_prob);
    }

    return changed;
}

bool update_reticulation_probs_linked(AnnotatedNetwork &ann_network, const std::vector<std::vector<DisplayedTreeData>>& displayed_trees_all) {
    bool changed = false;
    unsigned int n_sites_total = 0;
    std::vector<size_t> total_taken(ann_network.network.num_reticulations(), 0);

    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        size_t n_sites = displayed_trees_all[partition_idx][0].tree_persite_logl.size();
        n_sites_total += ann_network.fake_treeinfo->partitions[partition_idx]->pattern_weight_sum;
        std::vector<double> best_persite_logl(n_sites, -std::numeric_limits<double>::infinity());
        std::vector<size_t> best_tree_idx(n_sites, 0);
        for (const auto &tree : displayed_trees_all[partition_idx]) {
            for (size_t i = 0; i < tree.tree_persite_logl.size(); ++i) {
                if (tree.tree_persite_logl[i] > best_persite_logl[i]) {
                    best_persite_logl[i] = tree.tree_persite_logl[i];
                    best_tree_idx[i] = tree.tree_idx;
                }
            }
        }
        for (size_t i = 0; i < n_sites; ++i) {
            // find the reticulation nodes that have taken their first parent in the best tree
            unsigned int site_weight = ann_network.fake_treeinfo->partitions[partition_idx]->pattern_weights[i];
            for (size_t j = 0; j < ann_network.network.num_reticulations(); ++j) {
                if (!(best_tree_idx[i] & (1 << j))) {
                    total_taken[j] += site_weight;
                }
            }
        }
    }

    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        size_t pmatrix_index = getReticulationFirstParentPmatrixIndex(ann_network.network.reticulation_nodes[i]);
        double old_prob = ann_network.branch_probs[0][pmatrix_index];
        double new_prob = (double) total_taken[i] / n_sites_total;
        ann_network.branch_probs[0][pmatrix_index] = new_prob;
        changed |= (old_prob != new_prob);
    }

    return changed;
}

double recompute_network_logl_average_trees(AnnotatedNetwork &ann_network, std::vector<std::vector<DisplayedTreeData>>& displayed_trees_all) {
    mpfr::mpreal network_logl = 0.0;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
        mpfr::mpreal partition_lh = 0.0;
        for (auto& tree : displayed_trees_all[partition_idx]) {
            // update tree logprob with the new reticulation probabilities
            tree.tree_logprob = displayed_tree_logprob(ann_network, tree.tree_idx, partition_idx);
            assert(tree.tree_logl != 0);
            partition_lh += mpfr::exp(tree.tree_logprob) * mpfr::exp(tree.tree_logl);
        }
        fake_treeinfo.partition_loglh[partition_idx] = mpfr::log(partition_lh).toDouble();
        network_logl += mpfr::log(partition_lh);
    }
    ann_network.old_logl = network_logl.toDouble();
    return ann_network.old_logl;
}

double recompute_network_logl_best_tree(AnnotatedNetwork &ann_network, std::vector<std::vector<DisplayedTreeData>>& displayed_trees_all) {
    double network_logl = 0.0;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
        double partition_logl = std::numeric_limits<double>::min();
        for (auto& tree : displayed_trees_all[partition_idx]) {
            // update tree logprob with the new reticulation probabilities
            tree.tree_logprob = displayed_tree_logprob(ann_network, tree.tree_idx, partition_idx);
            assert(tree.tree_logl != 0);
            partition_logl = std::max(partition_logl, tree.tree_logprob + tree.tree_logl);
        }
        fake_treeinfo.partition_loglh[partition_idx] = partition_logl;
        network_logl += partition_logl;
    }
    ann_network.old_logl = network_logl;
    return ann_network.old_logl;
}

double recompute_network_logl(AnnotatedNetwork &ann_network, std::vector<std::vector<DisplayedTreeData>>& displayed_trees_all) {
    if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
        return recompute_network_logl_average_trees(ann_network, displayed_trees_all);
    } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
        return recompute_network_logl_best_tree(ann_network, displayed_trees_all);
    }
}

double computeLoglikelihood_new(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices,
        bool update_reticulation_probs) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    if (incremental & fake_treeinfo.clv_valid[0][network.root->clv_index]
            & !update_reticulation_probs) {
        return ann_network.old_logl;
    }

    mpfr::mpreal network_logl = 0.0;

    assert(!ann_network.branch_probs.empty());
    setup_pmatrices(ann_network, incremental, update_pmatrices);
    const int old_active_partition = fake_treeinfo.active_partition;
    fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    std::vector<bool> touched(network.nodes.size(), false);
    std::vector<Node*> parent = grab_current_node_parents(network);
    bool reticulation_probs_changed = false;
    std::vector<std::vector<DisplayedTreeData>> displayed_trees_all;

    for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
        fake_treeinfo.active_partition = partition_idx;
        std::vector<bool> *touched_ptr = nullptr;
        if (partition_idx == 0) {
            touched_ptr = &touched;
        }
        std::vector<DisplayedTreeData> displayed_trees = process_partition_new(ann_network, partition_idx, incremental, parent, touched_ptr);

        if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
            mpfr::mpreal partition_lh = 0.0;
            for (const auto& tree : displayed_trees) {
                assert(tree.tree_logl != 0);
                partition_lh += mpfr::exp(tree.tree_logprob) * mpfr::exp(tree.tree_logl);
            }
            fake_treeinfo.partition_loglh[partition_idx] = mpfr::log(partition_lh).toDouble();
            network_logl += mpfr::log(partition_lh);
        } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
            double partition_logl = -std::numeric_limits<double>::infinity();
            for (const auto& tree : displayed_trees) {
                assert(tree.tree_logl != 0);
                partition_logl = std::max(partition_logl, tree.tree_logprob + tree.tree_logl);
            }
            fake_treeinfo.partition_loglh[partition_idx] = partition_logl;
            network_logl += partition_logl;
        }

        if (update_reticulation_probs) {
            displayed_trees_all.emplace_back(displayed_trees);
            if (unlinked_mode) {
                reticulation_probs_changed |= update_reticulation_probs_unlinked(ann_network, displayed_trees, partition_idx);
            }
        }
    }
    if (update_reticulation_probs && !unlinked_mode) {
        reticulation_probs_changed = update_reticulation_probs_linked(ann_network, displayed_trees_all);
    }

    ann_network.old_logl = network_logl.toDouble();

    if (reticulation_probs_changed) {
        recompute_network_logl(ann_network, displayed_trees_all);
    }
    //std::cout << "network logl: " << ann_network.old_logl << "\n";
    return ann_network.old_logl;
}


mpfr::mpreal displayed_tree_nonblob_prob(AnnotatedNetwork &ann_network, size_t tree_index,
        size_t partition_index) {
    Network &network = ann_network.network;
    setReticulationParents(network, tree_index);
    mpfr::mpreal logProb = 0;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        size_t active_pmatrix_idx = getReticulationActiveParentPmatrixIndex(
                network.reticulation_nodes[i]);
        mpfr::mpreal prob = ann_network.branch_probs[partition_index][active_pmatrix_idx];
        logProb += mpfr::log(prob);
    }
    return mpfr::exp(logProb);
}

/**
 * Compute network loglikelihood by converting each displayed tree into a pll_utree_t and calling raxml loglikelihood on it.
 * Then taking the weighted sum of the displayed tree likelihoods (using exp) for the network likelihood, and returning th log of the network likelihood.
 * Using arbitrary-precision floating point operations for aggregating over the displayed tree loglikelihoods.
 * This naive implementation assumes a single partition in the MSA. It also doesn't do model optimization.
 **/
double computeLoglikelihoodNaiveUtree(AnnotatedNetwork &ann_network, int incremental,
        int update_pmatrices, std::vector<double> *treewise_logl) {
    (void) incremental;
    (void) update_pmatrices;
    Network &network = ann_network.network;
    RaxmlWrapper wrapper(ann_network.options);
    assert(wrapper.num_partitions() == 1);

    size_t n_trees = 1 << network.num_reticulations();
    mpfr::mpreal network_l = 0.0;
    // Iterate over all displayed trees
    for (size_t i = 0; i < n_trees; ++i) {
        mpfr::mpreal tree_prob = displayed_tree_nonblob_prob(ann_network, i, 0);
        if (tree_prob == 0.0) {
            continue;
        }
        pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, i);
        TreeInfo *displayedTreeinfo = wrapper.createRaxmlTreeinfo(displayed_tree);
        mpfr::mpreal tree_logl = displayedTreeinfo->loglh(0);
        delete displayedTreeinfo;
        if (treewise_logl) {
            treewise_logl->emplace_back(tree_logl.toDouble());
        }
        assert(tree_logl != -std::numeric_limits<double>::infinity());
        network_l += mpfr::exp(tree_logl) * tree_prob;
    }

    return mpfr::log(network_l).toDouble();
}


double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices,
        bool update_reticulation_probs) {
    return computeLoglikelihood_new(ann_network, incremental, update_pmatrices, update_reticulation_probs);
}

}
