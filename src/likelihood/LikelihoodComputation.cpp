/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../graph/Node.hpp"
#include "../DebugPrintFunctions.hpp"
#include "Operations.hpp"
#include "DisplayedTreeData.hpp"

#include <cassert>
#include <cmath>
#include "mpreal.h"

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

DisplayedTreeData& findMatchingDisplayedTree(AnnotatedNetwork& ann_network, const std::vector<ReticulationState>& reticulationChoices, NodeDisplayedTreeData& data) {
    DisplayedTreeData* tree = nullptr;
    assert(reticulationChoices.size() == ann_network.options.max_reticulations);
    
    size_t n_good = 0;
    for (size_t i = 0; i < data.num_active_displayed_trees; ++i) {
        assert(data.displayed_trees[i].reticulationChoices.size() == ann_network.options.max_reticulations);
        if (reticulationChoicesCompatible(reticulationChoices, data.displayed_trees[i].reticulationChoices)) {
            n_good++;
            tree = &data.displayed_trees[i];
        }
    }
    if (n_good == 1) {
        return *tree;
    } else if (n_good > 1) {
        std::cout << exportDebugInfo(ann_network) << "\n";
        for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
            std::cout << "displayed trees stored at node " << i << ":\n";
            for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[0][i].num_active_displayed_trees; ++j) {
                printReticulationChoices(ann_network.pernode_displayed_tree_data[0][i].displayed_trees[j].reticulationChoices);
            }
        }
        throw std::runtime_error("Found multiple suitable trees");
    } else { // n_good == 0
        throw std::runtime_error("Found no suitable displayed tree");
    }
}

Node* findFirstNodeWithTwoActiveChildren(AnnotatedNetwork& ann_network, const std::vector<ReticulationState>& reticulationChoices, Node* oldRoot) {
    for (size_t i = 0; i < reticulationChoices.size(); ++i) { // apply the reticulation choices
        setReticulationState(ann_network, i, reticulationChoices[i]);
    }

    Node* displayed_tree_root = nullptr;
    collect_dead_nodes(ann_network.network, oldRoot->clv_index, &displayed_tree_root);
    return displayed_tree_root;
}

void computeDisplayedTreeLoglikelihood(AnnotatedNetwork& ann_network, unsigned int partition_idx, DisplayedTreeData& treeAtRoot, Node* actRoot) {
    Node* displayed_tree_root = findFirstNodeWithTwoActiveChildren(ann_network, treeAtRoot.reticulationChoices, actRoot);
    DisplayedTreeData& treeWithoutDeadPath = findMatchingDisplayedTree(ann_network, treeAtRoot.reticulationChoices, ann_network.pernode_displayed_tree_data[partition_idx][displayed_tree_root->clv_index]);

    double* parent_clv = treeWithoutDeadPath.clv_vector;
    unsigned int* parent_scaler = treeWithoutDeadPath.scale_buffer;

    pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
    double tree_logl = pll_compute_root_loglikelihood(partition, displayed_tree_root->clv_index, parent_clv, parent_scaler, ann_network.fake_treeinfo->param_indices[partition_idx], nullptr);

    //std::cout << "computed tree logl at node " << displayed_tree_root->clv_index << ": " << tree_logl << "\n";

    treeAtRoot.tree_logl = tree_logl;
    treeAtRoot.tree_logl_valid = true;
    treeAtRoot.tree_logprob = computeReticulationChoicesLogProb(treeAtRoot.reticulationChoices, ann_network.reticulation_probs);
    treeAtRoot.tree_logprob_valid = true;
}

void iterateOverClv(double* clv, ClvRangeInfo& clvInfo) {
    if (!clv) return;
    std::cout << "Iterating over " << clvInfo.inner_clv_num_entries << " clv entries.\n";
    for (size_t i = 0; i < clvInfo.inner_clv_num_entries; ++i) {
        std::cout << clv[i] << " ";
    }
    std::cout << "\n";
}

void iterateOverScaler(unsigned int* scaler, ScaleBufferRangeInfo& scalerInfo) {
    if (!scaler) return;
    std::cout << "Iterating over " << scalerInfo.scaler_size << " scaler entries.\n";
    for (size_t i = 0; i < scalerInfo.scaler_size; ++i) {
        std::cout << scaler[i] << " ";
    }
    std::cout << "\n";
}

unsigned int processNodeImprovedSingleChild(AnnotatedNetwork& ann_network, unsigned int partition_idx, ClvRangeInfo &clvInfo, ScaleBufferRangeInfo &scaleBufferInfo, Node* node, Node* child) {
    unsigned int num_trees_added = 0;
    pll_operation_t op = buildOperationInternal(ann_network.network, node, child, nullptr, ann_network.network.nodes.size(), ann_network.network.edges.size());
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[partition_idx][node->clv_index];
    NodeDisplayedTreeData& displayed_trees_child = ann_network.pernode_displayed_tree_data[partition_idx][child->clv_index];
    size_t fake_clv_index = ann_network.network.nodes.size();
    pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];

    std::vector<ReticulationState> restrictions(ann_network.options.max_reticulations, ReticulationState::DONT_CARE);
    if (child->getType() == NodeType::RETICULATION_NODE) {
        if (node == getReticulationFirstParent(ann_network.network, child)) {
            restrictions[child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
        } else {
            restrictions[child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
        }
    }
    std::vector<Node*> children = getChildren(ann_network.network, node);
    if (children.size() == 2) {
        Node* other_child = getOtherChild(ann_network.network, node, child);
        assert(other_child->getType() == NodeType::RETICULATION_NODE);
        if (node == getReticulationFirstParent(ann_network.network, other_child)) {
            restrictions[other_child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
        } else {
            restrictions[other_child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
        }
    }

    for (size_t i = 0; i < displayed_trees_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& childTree = displayed_trees_child.displayed_trees[i];
        if (!reticulationChoicesCompatible(childTree.reticulationChoices, restrictions)) {
            continue;
        }
        displayed_trees.add_displayed_tree(clvInfo, scaleBufferInfo, ann_network.options.max_reticulations);
        DisplayedTreeData& tree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];
        double* parent_clv = tree.clv_vector;
        unsigned int* parent_scaler = tree.scale_buffer;
        double* left_clv = childTree.clv_vector;
        unsigned int* left_scaler = childTree.scale_buffer;
        double* right_clv = partition->clv[fake_clv_index];
        unsigned int* right_scaler = nullptr;

        pll_update_partials_single(partition, &op, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
        tree.reticulationChoices = combineReticulationChoices(childTree.reticulationChoices, restrictions);

        if (node == ann_network.network.root) { // if we are at the root node, we also need to compute loglikelihood
            computeDisplayedTreeLoglikelihood(ann_network, partition_idx, tree, node);
        } /*else { // this is just for debug
            computeDisplayedTreeLoglikelihood(ann_network, partition_idx, displayed_trees.displayed_trees[i], node);
        }*/
    }
    num_trees_added = displayed_trees_child.num_active_displayed_trees;
    return num_trees_added;
}

unsigned int processNodeImprovedTwoChildren(AnnotatedNetwork& ann_network, unsigned int partition_idx, ClvRangeInfo &clvInfo, ScaleBufferRangeInfo &scaleBufferInfo, Node* node, Node* left_child, Node* right_child) {
    unsigned int num_trees_added = 0;
    
    pll_operation_t op = buildOperationInternal(ann_network.network, node, left_child, right_child, ann_network.network.nodes.size(), ann_network.network.edges.size());
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[partition_idx][node->clv_index];
    NodeDisplayedTreeData& displayed_trees_left_child = ann_network.pernode_displayed_tree_data[partition_idx][left_child->clv_index];
    NodeDisplayedTreeData& displayed_trees_right_child = ann_network.pernode_displayed_tree_data[partition_idx][right_child->clv_index];
    pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];

    std::vector<ReticulationState> restrictions(ann_network.options.max_reticulations);
    if (left_child->getType() == NodeType::RETICULATION_NODE) {
        if (node == getReticulationFirstParent(ann_network.network, left_child)) {
            restrictions[left_child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
        } else {
            restrictions[left_child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
        }
    }
    if (right_child->getType() == NodeType::RETICULATION_NODE) {
        if (node == getReticulationFirstParent(ann_network.network, right_child)) {
            restrictions[right_child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
        } else {
            restrictions[right_child->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
        }
    }

    // left child and right child are not always about different reticulations... It can be that one reticulation affects both children.
    // It can even happen that there is a displayed tree for one child, that has no matching displaying tree on the other side (in terms of chosen reticulations). In this case, we have a dead node situation...
    std::vector<bool> rightTreeUsed(displayed_trees_right_child.num_active_displayed_trees, false);
    for (size_t i = 0; i < displayed_trees_left_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& leftTree = displayed_trees_left_child.displayed_trees[i];
        if (!reticulationChoicesCompatible(leftTree.reticulationChoices, restrictions)) {
            continue;
        }

        size_t n_compatible = 0;
        for (size_t j = 0; j < displayed_trees_right_child.num_active_displayed_trees; ++j) {
            DisplayedTreeData& rightTree = displayed_trees_right_child.displayed_trees[j];
            if (!reticulationChoicesCompatible(rightTree.reticulationChoices, restrictions)) {
                continue;
            }

            if (reticulationChoicesCompatible(leftTree.reticulationChoices, rightTree.reticulationChoices)) {
                rightTreeUsed[j] = true;
                n_compatible++;

                displayed_trees.add_displayed_tree(clvInfo, scaleBufferInfo, ann_network.options.max_reticulations);
                num_trees_added++;
                DisplayedTreeData& newDisplayedTree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];

                double* parent_clv = newDisplayedTree.clv_vector;
                unsigned int* parent_scaler = newDisplayedTree.scale_buffer;
                double* left_clv = leftTree.clv_vector;
                unsigned int* left_scaler = leftTree.scale_buffer;
                double* right_clv = rightTree.clv_vector;
                unsigned int* right_scaler = rightTree.scale_buffer;
                pll_update_partials_single(partition, &op, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
                newDisplayedTree.reticulationChoices = combineReticulationChoices(leftTree.reticulationChoices, rightTree.reticulationChoices);
                newDisplayedTree.reticulationChoices = combineReticulationChoices(newDisplayedTree.reticulationChoices, restrictions);

                if (node == ann_network.network.root) { // if we are at the root node, we also need to compute loglikelihood
                    computeDisplayedTreeLoglikelihood(ann_network, partition_idx, newDisplayedTree, node);
                }/* else { // this is just for debug
                    computeDisplayedTreeLoglikelihood(ann_network, partition_idx, newDisplayedTree, node);
                }*/
                
            }
        }
        if (n_compatible == 0) {
            // left displayed tree, right child is dead node
            if ((left_child->getType() != NodeType::RETICULATION_NODE) && (right_child->getType() != NodeType::RETICULATION_NODE)) {
                num_trees_added += processNodeImprovedSingleChild(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, left_child);
            }
        }
    }

    if ((left_child->getType() != NodeType::RETICULATION_NODE) && (right_child->getType() != NodeType::RETICULATION_NODE)) {
        for (size_t j = 0; j < displayed_trees_right_child.num_active_displayed_trees; ++j) {
            if (!rightTreeUsed[j]) {
                // right displayed tree, left child is dead node
                num_trees_added += processNodeImprovedSingleChild(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, right_child);
            }
        }
    }

    return num_trees_added;
}

void processNodeImproved(AnnotatedNetwork& ann_network, unsigned int partition_idx, int incremental, ClvRangeInfo &clvInfo, ScaleBufferRangeInfo &scaleBufferInfo, Node* node) {
    if (node->clv_index < ann_network.network.num_tips()) {
        //assert(ann_network.fake_treeinfo->clv_valid[partition_idx][node->clv_index]);
        return;
    }
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[partition_idx][node->clv_index];
    if (incremental && ann_network.fake_treeinfo->clv_valid[partition_idx][node->clv_index]) {
        return;
    }

    displayed_trees.num_active_displayed_trees = 0;

    std::vector<Node*> children = getChildren(ann_network.network, node);
    if (children.size() == 1) { // we are at a reticulation node
        assert(node->getType() == NodeType::RETICULATION_NODE);
        processNodeImprovedSingleChild(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, children[0]);
    } else {   
        Node* left_child = children[0];
        Node* right_child = children[1];
        bool left_child_reticulation = (left_child->getType() == NodeType::RETICULATION_NODE);
        bool right_child_reticulation = (right_child->getType() == NodeType::RETICULATION_NODE);

        for (int ignore_left_child = 0; ignore_left_child <= left_child_reticulation; ++ignore_left_child) {
            for (int ignore_right_child = 0; ignore_right_child <= right_child_reticulation; ++ignore_right_child) {
                if ((ignore_left_child == 1) && (ignore_right_child == 1)) { // no child
                    continue;  // We handle dead nodes by not storing any trees in them.
                }
                if (ignore_left_child) {
                    processNodeImprovedSingleChild(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, right_child);
                } else if (ignore_right_child) {
                    processNodeImprovedSingleChild(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, left_child);
                } else { // take both children
                    processNodeImprovedTwoChildren(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, left_child, right_child);
                }
            }
        }
    }

    ann_network.fake_treeinfo->clv_valid[partition_idx][node->clv_index] = 1;
}

void processPartitionImproved(AnnotatedNetwork& ann_network, unsigned int partition_idx, int incremental) {
    //std::cout << "\nNEW PROCESS PARTITION_IMPROVED!!!\n";
    std::vector<bool> seen(ann_network.network.num_nodes(), false);
    ClvRangeInfo clvInfo = get_clv_range(ann_network.fake_treeinfo->partitions[partition_idx]);
    ScaleBufferRangeInfo scaleBufferInfo = get_scale_buffer_range(ann_network.fake_treeinfo->partitions[partition_idx]);
    
    for (size_t i = 0; i < ann_network.travbuffer.size(); ++i) {
        Node* actNode = ann_network.travbuffer[i];
        processNodeImproved(ann_network, partition_idx, incremental, clvInfo, scaleBufferInfo, actNode);
    }
}

bool reuseOldDisplayedTreesCheck(AnnotatedNetwork& ann_network, int incremental) {
    if (!incremental) {
        return false;
    }
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool all_clvs_valid = true;
    for (size_t i = 0; i < fake_treeinfo.partition_count; ++i) {
        all_clvs_valid &= fake_treeinfo.clv_valid[i][ann_network.network.root->clv_index];
    }
    return all_clvs_valid;
}

double computeLoglikelihoodImproved(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    const Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool reuse_old_displayed_trees = reuseOldDisplayedTreesCheck(ann_network, incremental);
    mpfr::mpreal network_logl = 0.0;
    if (reuse_old_displayed_trees) {
        for (size_t p = 0; p < fake_treeinfo.partition_count; ++p) { // TODO: Why is this needed here?
            std::vector<DisplayedTreeData>& displayed_root_trees = ann_network.pernode_displayed_tree_data[p][network.root->clv_index].displayed_trees;
            size_t n_trees = ann_network.pernode_displayed_tree_data[p][network.root->clv_index].num_active_displayed_trees;
            for (size_t t = 0; t < n_trees; ++t) {
                assert(displayed_root_trees[t].tree_logl_valid == true);
                displayed_root_trees[t].tree_logprob = computeReticulationChoicesLogProb(displayed_root_trees[t].reticulationChoices, ann_network.reticulation_probs);
                displayed_root_trees[t].tree_logprob_valid = true;
            }
        }
    } else {
        fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
        setup_pmatrices(ann_network, incremental, update_pmatrices);
        for (size_t p = 0; p < fake_treeinfo.partition_count; ++p) {
            processPartitionImproved(ann_network, p, incremental);
        }
    }

    for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
        fake_treeinfo.active_partition = partition_idx;
        std::vector<DisplayedTreeData>& displayed_root_trees = ann_network.pernode_displayed_tree_data[partition_idx][network.root->clv_index].displayed_trees;
        size_t n_trees = ann_network.pernode_displayed_tree_data[partition_idx][network.root->clv_index].num_active_displayed_trees;

        if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
            mpfr::mpreal partition_lh = 0.0;
            for (size_t tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
                DisplayedTreeData& tree = displayed_root_trees[tree_idx];
                assert(tree.tree_logl_valid);
                assert(tree.tree_logprob_valid);
                assert(tree.tree_logl != 0);
                assert(tree.tree_logl != -std::numeric_limits<double>::infinity());
                //std::cout << "tree " << tree.tree_idx << " logl: " << tree.tree_logl << "\n";
                if (tree.tree_logprob != std::numeric_limits<double>::infinity()) {
                    partition_lh += mpfr::exp(tree.tree_logprob) * mpfr::exp(tree.tree_logl);
                }
            }
            double partition_logl = mpfr::log(partition_lh).toDouble();
            fake_treeinfo.partition_loglh[partition_idx] = partition_logl;
            network_logl += mpfr::log(partition_lh);
        } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
            double partition_logl = -std::numeric_limits<double>::infinity();
            for (size_t tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
                DisplayedTreeData& tree = displayed_root_trees[tree_idx];
                if (!tree.tree_logl_valid) {
                    std::cout << "tree_idx: " << tree_idx << "\n";
                    std::cout << "n_trees: " << n_trees << "\n";
                    std::cout << "displayed trees stored at node " << ann_network.network.root->clv_index << ":\n";
                    for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[0][ann_network.network.root->clv_index].num_active_displayed_trees; ++j) {
                        printReticulationChoices(ann_network.pernode_displayed_tree_data[0][ann_network.network.root->clv_index].displayed_trees[j].reticulationChoices);
                    }
                    throw std::runtime_error("invalid tree logl");
                }
                assert(tree.tree_logl_valid);
                assert(tree.tree_logprob_valid);
                assert(tree.tree_logl != 0);
                assert(tree.tree_logl != -std::numeric_limits<double>::infinity());
                //std::cout << "tree " << tree_idx << " logl: " << tree.tree_logl << "\n";
                //std::cout << "tree " << tree_idx << " logprob: " << tree.tree_logprob << "\n";
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
    fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    //std::cout << "network logl: " << network_logl.toDouble() << "\n";
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
    //return computeLoglikelihood_new(ann_network, incremental, update_pmatrices);
    return computeLoglikelihoodImproved(ann_network, incremental, update_pmatrices);
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
