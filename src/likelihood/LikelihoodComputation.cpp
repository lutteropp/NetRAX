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

mpfr::mpreal displayed_tree_prob_blobs(AnnotatedNetwork &ann_network, size_t megablob_idx) {
    BlobInformation &blobInfo = ann_network.blobInfo;
    mpfr::mpreal logProb = 0;
    for (size_t i = 0; i < blobInfo.reticulation_nodes_per_megablob[megablob_idx].size(); ++i) {
        mpfr::mpreal prob = getReticulationActiveProb(ann_network, blobInfo.reticulation_nodes_per_megablob[megablob_idx][i]);
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
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    if (update_pmatrices) {
        pllmod_treeinfo_update_prob_matrices(&fake_treeinfo, !incremental);
    }
}


double displayed_tree_logprob(AnnotatedNetwork &ann_network, size_t tree_index) {
    Network &network = ann_network.network;
    setReticulationParents(network, tree_index);
    mpfr::mpreal logProb = 0;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        mpfr::mpreal prob = getReticulationActiveProb(ann_network, network.reticulation_nodes[i]);
        if (prob == 0.0) {
            return std::numeric_limits<double>::infinity();
        }
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
    double tree_logprob = displayed_tree_logprob(ann_network, tree_idx);
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
            std::cout << exportDebugInfo(ann_network) << "\n";
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
        const std::vector<Node*> &parent) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;

    // just for debug: invalidate all pmatrix and clv indices
    /*for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
        ann_network.fake_treeinfo->clv_valid[partition_idx][i] = 0;
    }
    for (size_t i = 0; i < ann_network.network.edges.size(); ++i) {
        ann_network.fake_treeinfo->pmatrix_valid[partition_idx][i] = 0;
    }*/

    fake_treeinfo.active_partition = partition_idx;
    setup_pmatrices(ann_network, incremental, true);

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

double computeLoglikelihood_new(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    //incremental = 0;
    //update_pmatrices = 1;

    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool reuse_old_displayed_trees = false;
    if (incremental) {
        bool all_clvs_valid = true;
        for (size_t i = 0; i < fake_treeinfo.partition_count; ++i) {
            all_clvs_valid &= fake_treeinfo.clv_valid[i][network.root->clv_index];
        }
        if (all_clvs_valid) {
            reuse_old_displayed_trees = true;
        }
    }

    mpfr::mpreal network_logl = 0.0;

    fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    setup_pmatrices(ann_network, incremental, update_pmatrices);
    std::vector<Node*> parent = grab_current_node_parents(network);

    std::vector<std::vector<DisplayedTreeData> > all_displayed_trees(fake_treeinfo.partition_count);
    if (reuse_old_displayed_trees) {
        all_displayed_trees = ann_network.old_displayed_trees;
    } else {
        for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
            fake_treeinfo.active_partition = partition_idx;
            std::vector<DisplayedTreeData> displayed_trees = process_partition_new(ann_network, partition_idx, incremental, parent);
            for (size_t i = 0; i < displayed_trees.size(); ++i) {
                all_displayed_trees[partition_idx].emplace_back(displayed_trees[i]);
            }
        }
    }

    for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
        fake_treeinfo.active_partition = partition_idx;
        if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
            mpfr::mpreal partition_lh = 0.0;
            for (const auto& tree : all_displayed_trees[partition_idx]) {
                assert(tree.tree_logl != 0);
                //std::cout << "tree " << tree.tree_idx << " logl: " << tree.tree_logl << "\n";
                if (tree.tree_logprob != std::numeric_limits<double>::infinity()) {
                    partition_lh += mpfr::exp(tree.tree_logprob) * mpfr::exp(tree.tree_logl);
                }
            }
            fake_treeinfo.partition_loglh[partition_idx] = mpfr::log(partition_lh).toDouble();
            network_logl += mpfr::log(partition_lh);
        } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
            double partition_logl = -std::numeric_limits<double>::infinity();
            for (const auto& tree : all_displayed_trees[partition_idx]) {
                assert(tree.tree_logl != 0);
                //std::cout << "tree " << tree.tree_idx << " logl: " << tree.tree_logl << "\n";
                //std::cout << "tree " << tree.tree_idx << " logprob: " << tree.tree_logprob << "\n";
                if (tree.tree_logprob != std::numeric_limits<double>::infinity()) {
                    //std::cout << "tree " << tree.tree_idx << " prob: " << mpfr::exp(tree.tree_logprob) << "\n";
                    partition_logl = std::max(partition_logl, tree.tree_logprob + tree.tree_logl);
                }
            }
            fake_treeinfo.partition_loglh[partition_idx] = partition_logl;
            //std::cout << "partiion " << partition_idx << " logl: " << partition_logl << "\n";
            network_logl += partition_logl;
        }
    }

    //std::cout << "total_logl: " << network_logl << "\n";

    ann_network.old_displayed_trees = all_displayed_trees;
    for (size_t i = 0; i < fake_treeinfo.partition_count; ++i) {
        fake_treeinfo.clv_valid[i][network.root->clv_index] = 1;
    }
    //std::cout << "network logl: " << ann_network.old_logl << "\n";
    return network_logl.toDouble();
}


mpfr::mpreal displayed_tree_nonblob_prob(AnnotatedNetwork &ann_network, size_t tree_index) {
    Network &network = ann_network.network;
    setReticulationParents(network, tree_index);
    mpfr::mpreal logProb = 0;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        mpfr::mpreal prob = getReticulationActiveProb(ann_network, network.reticulation_nodes[i]);
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
    size_t num_partitions = ann_network.fake_treeinfo->partition_count;
    size_t num_trees = (1 << ann_network.network.num_reticulations());

    std::vector<Model> partition_models(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
        assign(partition_models[i], ann_network.fake_treeinfo->partitions[i]);
    }

    assert(ann_network.options.brlen_linkage != PLLMOD_COMMON_BRLEN_UNLINKED);
    // ensure the network edge lengths are up-to-date (needed for displayed tree to utree)
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        ann_network.network.edges_by_index[i]->length = ann_network.fake_treeinfo->branch_lengths[0][i];
    }

    std::vector<std::vector<double> > tree_logl_per_partition(num_partitions, std::vector<double>(num_trees, -std::numeric_limits<double>::infinity()));
    std::vector<double> tree_logprob(num_trees, 0.0);

    // Iterate over all displayed trees
    for (size_t i = 0; i < num_trees; ++i) {
        tree_logprob[i] =  displayed_tree_logprob(ann_network, i);
        if (tree_logprob[i] == std::numeric_limits<double>::infinity()) {
            continue;
        }
        pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, i);
        TreeInfo *displayedTreeinfo = wrapper.createRaxmlTreeinfo(displayed_tree);
        for (size_t p = 0; p < num_partitions; ++p) {
            displayedTreeinfo->model(p, partition_models[p]);
        }

        double tree_logl = displayedTreeinfo->loglh(0);
        std::vector<double> partition_tree_logl(num_partitions);
        for (size_t p = 0; p < num_partitions; ++p) {
            tree_logl_per_partition[p][i] = displayedTreeinfo->pll_treeinfo().partition_loglh[p];
        }
        delete displayedTreeinfo;

        if (treewise_logl) {
            treewise_logl->emplace_back(tree_logl);
        }
        assert(tree_logl != -std::numeric_limits<double>::infinity());
    }

    mpfr::mpreal network_logl = 0.0;
    for (size_t partition_idx = 0; partition_idx < num_partitions; ++partition_idx) {
        double partition_logl = -std::numeric_limits<double>::infinity();
        if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
            mpfr::mpreal partition_lh = 0.0;
            for (size_t i = 0; i < num_trees; ++i) {
                if (tree_logprob[i] != std::numeric_limits<double>::infinity()) {
                    partition_lh += mpfr::exp(tree_logl_per_partition[partition_idx][i]) * mpfr::exp(tree_logprob[i]);
                }
            }
            partition_logl = mpfr::log(partition_lh).toDouble();
        } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
            for (size_t i = 0; i < num_trees; ++i) {
                if (tree_logprob[i] != std::numeric_limits<double>::infinity()) {
                    partition_logl = std::max(partition_logl, tree_logl_per_partition[partition_idx][i] + tree_logprob[i]);
                }
            }
            //std::cout << "partiion " << partition_idx << " logl: " << partition_logl << "\n";
        }
        network_logl += partition_logl;
        ann_network.fake_treeinfo->partition_loglh[partition_idx] = partition_logl;
    }
    return network_logl.toDouble();
}


double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    //just for debug
    //incremental = 0;
    //update_pmatrices = 1;
    return computeLoglikelihood_new(ann_network, incremental, update_pmatrices);
    //return computeLoglikelihoodNaiveUtree(ann_network, incremental, update_pmatrices);
}

double aic(double logl, double k) {
    return -2 * logl + 2 * k;
}
double aicc(double logl, double k, double n) {
    return aic(logl, k) + (2*k*k + 2*k) / (n - k - 1);
}
double bic(double logl, double k, double n) {
    return -2 * logl + k * log(n);
}

size_t get_param_count(AnnotatedNetwork& ann_network) {
    Network &network = ann_network.network;

    size_t param_count = ann_network.total_num_model_parameters;
    // reticulation probs as free parameters
    param_count += ann_network.network.num_reticulations();
    if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        assert(ann_network.fake_treeinfo->partition_count > 1);
        param_count += ann_network.fake_treeinfo->partition_count * network.num_branches();
    } else { // branch lengths are shared among partitions
        param_count += network.num_branches();
        if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
            assert(ann_network.fake_treeinfo->partition_count > 1);
            // each partition can scale the branch lengths by its own scaling factor
            param_count += ann_network.fake_treeinfo->partition_count - 1;
        }
    }
    return param_count;
}

size_t get_sample_size(AnnotatedNetwork& ann_network) {
    return ann_network.total_num_sites * ann_network.network.num_tips();
}

double aic(AnnotatedNetwork &ann_network, double logl) {
    return aic(logl, get_param_count(ann_network));
}

double aicc(AnnotatedNetwork &ann_network, double logl) {
    return aicc(logl, get_param_count(ann_network), get_sample_size(ann_network));
}

double bic(AnnotatedNetwork &ann_network, double logl) {
    return bic(logl, get_param_count(ann_network), get_sample_size(ann_network));
}

/**
 * Computes the BIC-score of a given network. A smaller score is a better score.
 * 
 * @param ann_network The network.
 */
double scoreNetwork(AnnotatedNetwork &ann_network) {
    double logl = computeLoglikelihood(ann_network, 1, 1);
    return bic(ann_network, logl);
}

}
