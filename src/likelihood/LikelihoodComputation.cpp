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
#include "DisplayedTreeData.hpp"

#include <cassert>
#include <cmath>

namespace netrax {

void setup_pmatrices(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    if (update_pmatrices) {
        pllmod_treeinfo_update_prob_matrices(&fake_treeinfo, !incremental);
    }
}

pll_operation_t buildOperation(Network &network, Node *parent, Node *child1, Node *child2,
        size_t fake_clv_index, size_t fake_pmatrix_index) {
    pll_operation_t operation;
    assert(parent);
    operation.parent_clv_index = parent->clv_index;
    operation.parent_scaler_index = parent->scaler_index;
    if (child1) {
        operation.child1_clv_index = child1->clv_index;
        operation.child1_scaler_index = child1->scaler_index;
        operation.child1_matrix_index = getEdgeTo(network, child1, parent)->pmatrix_index;
    } else {
        operation.child1_clv_index = fake_clv_index;
        operation.child1_scaler_index = -1;
        operation.child1_matrix_index = fake_pmatrix_index;
    }
    if (child2) {
        operation.child2_clv_index = child2->clv_index;
        operation.child2_scaler_index = child2->scaler_index;
        operation.child2_matrix_index = getEdgeTo(network, child2, parent)->pmatrix_index;
    } else {
        operation.child2_clv_index = fake_clv_index;
        operation.child2_scaler_index = -1;
        operation.child2_matrix_index = fake_pmatrix_index;
    }
    return operation;
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

DisplayedTreeData& findMatchingDisplayedTree(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, NodeDisplayedTreeData& data) {
    DisplayedTreeData* tree = nullptr;
    
    size_t n_good = 0;
    for (size_t i = 0; i < data.num_active_displayed_trees; ++i) {
        if (reticulationConfigsCompatible(reticulationChoices, data.displayed_trees[i].treeLoglData.reticulationChoices)) {
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
            for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[i].num_active_displayed_trees; ++j) {
                printReticulationChoices(ann_network.pernode_displayed_tree_data[i].displayed_trees[j].treeLoglData.reticulationChoices);
            }
        }
        std::cout << "Reticulation first parents:\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << "reticulation node " << ann_network.network.reticulation_nodes[i]->clv_index << " has first parent " << getReticulationFirstParent(ann_network.network, ann_network.network.reticulation_nodes[i])->clv_index << "\n";
        }
        throw std::runtime_error("Found multiple suitable trees");
    } else { // n_good == 0
        throw std::runtime_error("Found no suitable displayed tree");
    }
}

Node* findFirstNodeWithTwoActiveChildren(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, Node* oldRoot) {
    // TODO: Make this work with direction-agnistic stuff (virtual rerooting)
    //throw std::runtime_error("TODO: Make this work with direction-agnistic stuff (virtual rerooting)");

    // all these reticulation choices led to the same tree, thus it is safe to simply use the first one for detecting which nodes to skip...
    for (size_t i = 0; i < reticulationChoices.configs[0].size(); ++i) { // apply the reticulation choices
        if (reticulationChoices.configs[0][i] != ReticulationState::DONT_CARE) {
            setReticulationState(ann_network.network, i, reticulationChoices.configs[0][i]);
        }
    }

    Node* displayed_tree_root = nullptr;
    collect_dead_nodes(ann_network.network, oldRoot->clv_index, &displayed_tree_root);
    return displayed_tree_root;
}

void computeDisplayedTreeLoglikelihood(AnnotatedNetwork& ann_network, DisplayedTreeData& treeAtRoot, Node* actRoot) {
    Node* displayed_tree_root = findFirstNodeWithTwoActiveChildren(ann_network, treeAtRoot.treeLoglData.reticulationChoices, actRoot);
    DisplayedTreeData& treeWithoutDeadPath = findMatchingDisplayedTree(ann_network, treeAtRoot.treeLoglData.reticulationChoices, ann_network.pernode_displayed_tree_data[displayed_tree_root->clv_index]);

    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        treeAtRoot.treeLoglData.tree_partition_logl[partition_idx] = 0.0;
        //skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
            continue;
        }
        double* parent_clv = treeWithoutDeadPath.clv_vector[partition_idx];
        unsigned int* parent_scaler = treeWithoutDeadPath.scale_buffer[partition_idx];

        pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
        std::vector<double> persite_logl(ann_network.fake_treeinfo->partitions[partition_idx]->sites, 0.0);
        double tree_logl = pll_compute_root_loglikelihood(partition, displayed_tree_root->clv_index, parent_clv, parent_scaler, ann_network.fake_treeinfo->param_indices[partition_idx], persite_logl.data());

        //std::cout << "computed tree logl at node " << displayed_tree_root->clv_index << ": " << tree_logl << "\n";

        if (tree_logl == -std::numeric_limits<double>::infinity()) {
            std::cout << "I am thread " << ParallelContext::local_proc_id() << " and I have gotten negative infinity for partition " << partition_idx << "\n";
            for (size_t i = 0; i < persite_logl.size(); ++i) {
                std::cout << "  persite_logl[" << i << "]: " << persite_logl[i] << "\n";
            }
            printClv(*ann_network.fake_treeinfo, ann_network.network.root->clv_index, parent_clv, partition_idx);
            assert(parent_clv);
        }

        assert(tree_logl != -std::numeric_limits<double>::infinity());
        assert(tree_logl <= 0.0);
        treeAtRoot.treeLoglData.tree_partition_logl[partition_idx] = tree_logl;
    }

    /* sum up likelihood from all threads */
    if (ann_network.fake_treeinfo->parallel_reduce_cb)
    {   
        std::cout << "I am thread " << ParallelContext::local_proc_id() << " and I have these partition loglikelihoods before Allreduce:\n";
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            std::cout << " partition " << p << " has logl: " << treeAtRoot.treeLoglData.tree_partition_logl[p] << "\n";
        }

        ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context,
                                    treeAtRoot.treeLoglData.tree_partition_logl.data(),
                                    ann_network.fake_treeinfo->partition_count,
                                    PLLMOD_COMMON_REDUCE_SUM);

        std::cout << "I am thread " << ParallelContext::local_proc_id() << " and I have these partition loglikelihoods after Allreduce:\n";
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            std::cout << " partition " << p << " has logl: " << treeAtRoot.treeLoglData.tree_partition_logl[p] << "\n";
        }

        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            assert(treeAtRoot.treeLoglData.tree_partition_logl[p] != -std::numeric_limits<double>::infinity());
            assert(treeAtRoot.treeLoglData.tree_partition_logl[p] <= 0.0);
        }
    }

    treeAtRoot.treeLoglData.tree_logl_valid = true;
    treeAtRoot.treeLoglData.tree_logprob = computeReticulationConfigLogProb(treeAtRoot.treeLoglData.reticulationChoices, ann_network.reticulation_probs);
    treeAtRoot.treeLoglData.tree_logprob_valid = true;
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

ReticulationConfigSet getRestrictionsToDismissNeighbor(AnnotatedNetwork& ann_network, Node* node, Node* neighbor) {
    ReticulationConfigSet res(ann_network.options.max_reticulations);
    std::vector<ReticulationState> restrictions(ann_network.options.max_reticulations, ReticulationState::DONT_CARE);
    assert(node);
    assert(neighbor);
    bool foundRestriction = false;
    if (node->getType() == NodeType::RETICULATION_NODE) {
        if (neighbor == getReticulationFirstParent(ann_network.network, node)) {
            restrictions[node->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
            foundRestriction = true;
        } else if (neighbor == getReticulationSecondParent(ann_network.network, node)) {
            restrictions[node->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
            foundRestriction = true;
        }
    }
    if (neighbor->getType() == NodeType::RETICULATION_NODE) {
        if (node == getReticulationFirstParent(ann_network.network, neighbor)) {
            restrictions[neighbor->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
            foundRestriction = true;
        } else if (node == getReticulationSecondParent(ann_network.network, neighbor)) {
            restrictions[neighbor->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
            foundRestriction = true;
        }
    }
    if (foundRestriction) {
        res.configs.emplace_back(restrictions);
    }
    return res;
}

ReticulationConfigSet getRestrictionsToTakeNeighbor(AnnotatedNetwork& ann_network, Node* node, Node* neighbor) {
    ReticulationConfigSet res(ann_network.options.max_reticulations);
    std::vector<ReticulationState> restrictions(ann_network.options.max_reticulations, ReticulationState::DONT_CARE);
    assert(node);
    assert(neighbor);
    if (node->getType() == NodeType::RETICULATION_NODE) {
        if (neighbor == getReticulationFirstParent(ann_network.network, node)) {
            restrictions[node->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
        } else if (neighbor == getReticulationSecondParent(ann_network.network, node)) {
            restrictions[node->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
        }
    }
    if (neighbor->getType() == NodeType::RETICULATION_NODE) {
        if (node == getReticulationFirstParent(ann_network.network, neighbor)) {
            restrictions[neighbor->getReticulationData()->reticulation_index] = ReticulationState::TAKE_FIRST_PARENT;
        } else if (node == getReticulationSecondParent(ann_network.network, neighbor)) {
            restrictions[neighbor->getReticulationData()->reticulation_index] = ReticulationState::TAKE_SECOND_PARENT;
        }
    }
    res.configs.emplace_back(restrictions);
    return res;
}

void add_displayed_tree(AnnotatedNetwork& ann_network, size_t clv_index) {
    NodeDisplayedTreeData& data = ann_network.pernode_displayed_tree_data[clv_index];
    data.num_active_displayed_trees++;

    if (data.num_active_displayed_trees > data.displayed_trees.size()) {
        assert(data.num_active_displayed_trees == data.displayed_trees.size() + 1);
        data.displayed_trees.emplace_back(DisplayedTreeData(ann_network.fake_treeinfo, ann_network.partition_clv_ranges, ann_network.partition_scale_buffer_ranges, ann_network.options.max_reticulations));
    } else { // zero out the clv vector and scale buffer
        for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
                continue;
            }
            assert(data.displayed_trees[data.num_active_displayed_trees-1].clv_vector[partition_idx]);
            memset(data.displayed_trees[data.num_active_displayed_trees-1].clv_vector[partition_idx], 0, ann_network.partition_clv_ranges[partition_idx].inner_clv_num_entries * sizeof(double));
            if (data.displayed_trees[data.num_active_displayed_trees-1].scale_buffer[partition_idx]) {
                memset(data.displayed_trees[data.num_active_displayed_trees-1].scale_buffer[partition_idx], 0, ann_network.partition_scale_buffer_ranges[partition_idx].scaler_size * sizeof(unsigned int));
            }
        }
    }
}

unsigned int processNodeImprovedSingleChild(AnnotatedNetwork& ann_network, Node* node, Node* child, const ReticulationConfigSet& extraRestrictions) {
    assert(node);
    assert(child);

    unsigned int num_trees_added = 0;
    pll_operation_t op = buildOperation(ann_network.network, node, child, nullptr, ann_network.network.nodes.size(), ann_network.network.edges.size());
    
    size_t fake_clv_index = ann_network.network.nodes.size();

    ReticulationConfigSet restrictionsSet = getRestrictionsToTakeNeighbor(ann_network, node, child);

    if (!extraRestrictions.configs.empty()) {
        restrictionsSet = combineReticulationChoices(restrictionsSet, extraRestrictions);
    }

    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[node->clv_index];
    NodeDisplayedTreeData& displayed_trees_child = ann_network.pernode_displayed_tree_data[child->clv_index];

    for (size_t i = 0; i < displayed_trees_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& childTree = displayed_trees_child.displayed_trees[i];
        if (!reticulationConfigsCompatible(childTree.treeLoglData.reticulationChoices, restrictionsSet)) {
            continue;
        }
        add_displayed_tree(ann_network, node->clv_index);
        DisplayedTreeData& tree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];
        for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
                continue;
            }
            pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
            double* parent_clv = tree.clv_vector[partition_idx];
            unsigned int* parent_scaler = tree.scale_buffer[partition_idx];
            double* left_clv = childTree.clv_vector[partition_idx];
            unsigned int* left_scaler = childTree.scale_buffer[partition_idx];
            double* right_clv = partition->clv[fake_clv_index];
            unsigned int* right_scaler = nullptr;
            assert(childTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], left_clv));
            assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], right_clv));
            pll_update_partials_single(partition, &op, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
            assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], parent_clv));
        }
        tree.treeLoglData.reticulationChoices = combineReticulationChoices(childTree.treeLoglData.reticulationChoices, restrictionsSet);
        //tree.treeLoglData.childrenTaken = {child};
    }
    num_trees_added = displayed_trees_child.num_active_displayed_trees;

    return num_trees_added;
}

ReticulationConfigSet deadNodeSettings(AnnotatedNetwork& ann_network, const NodeDisplayedTreeData& displayed_trees, Node* parent, Node* child) {
    // Return all configurations in which the node which the displayed trees belong to would have no displayed tree, and thus be a dead node
    ReticulationConfigSet res(ann_network.options.max_reticulations);
    std::vector<ReticulationState> reticulationChoicesVector(ann_network.options.max_reticulations);
    ReticulationConfigSet reticulationChoices(ann_network.options.max_reticulations);
    reticulationChoices.configs.emplace_back(reticulationChoicesVector);

    ReticulationConfigSet childNotTakenRestriction = getRestrictionsToDismissNeighbor(ann_network, parent, child);
    for (size_t i = 0; i < childNotTakenRestriction.configs.size(); ++i) {
        res.configs.emplace_back(childNotTakenRestriction.configs[i]);
    }
    ReticulationConfigSet childTakenRestriction = getRestrictionsToTakeNeighbor(ann_network, parent, child);

    if (ann_network.network.num_reticulations() > sizeof(size_t) * 8) {
        throw std::runtime_error("This implementation only works for <= sizeof(size_t)*8 reticulations");
    }
    size_t max_n_trees = (1 << ann_network.network.num_reticulations());
    for (size_t tree_idx = 0; tree_idx < max_n_trees; ++tree_idx) {
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            if (tree_idx & (1 << i)) {
                reticulationChoices.configs[0][i] = ReticulationState::TAKE_SECOND_PARENT;
            } else {
                reticulationChoices.configs[0][i] = ReticulationState::TAKE_FIRST_PARENT;
            }
        }

        if (!reticulationConfigsCompatible(reticulationChoices, childTakenRestriction)) {
            continue;
        }

        bool foundTree = false;

        for (size_t i = 0; i < displayed_trees.num_active_displayed_trees; ++i) {
            if (reticulationConfigsCompatible(reticulationChoices, displayed_trees.displayed_trees[i].treeLoglData.reticulationChoices)) {
                foundTree = true;
                break;
            }
        }
        if (!foundTree) {
            res.configs.emplace_back(reticulationChoices.configs[0]);
        }   
    }

    simplifyReticulationChoices(res);
    return res;
}

ReticulationConfigSet getReticulationChoicesThisOnly(AnnotatedNetwork& ann_network, const ReticulationConfigSet& this_tree_config, const ReticulationConfigSet& other_child_dead_settings, Node* parent, Node* this_child, Node* other_child) {
    // covers both dead children and reticulation children
    ReticulationConfigSet res(ann_network.options.max_reticulations);

    ReticulationConfigSet this_reachable_from_parent_restrictionSet = getRestrictionsToTakeNeighbor(ann_network, parent, this_child);
    ReticulationConfigSet restrictedConfig = combineReticulationChoices(this_tree_config, this_reachable_from_parent_restrictionSet);

    if (restrictedConfig.configs.empty()) { // easy case, this_tree isn't reachable anyway
        return res;
    }

    // Find all configurations where we can take restricted_config, but we cannot take any of the trees from displayed_trees_other

    // easy case: parent not being an active parent of other_child        
    ReticulationConfigSet other_not_reachable_from_parent_restrictionSet = getRestrictionsToDismissNeighbor(ann_network, parent, other_child);    
    ReticulationConfigSet combinedConfig = combineReticulationChoices(restrictedConfig, other_not_reachable_from_parent_restrictionSet);
    for (size_t i = 0; i < combinedConfig.configs.size(); ++i) {
        res.configs.emplace_back(combinedConfig.configs[i]);
    }

    ReticulationConfigSet other_reachable_from_parent_restrictionSet = getRestrictionsToTakeNeighbor(ann_network, parent, other_child);
    restrictedConfig = combineReticulationChoices(restrictedConfig, other_reachable_from_parent_restrictionSet);
    // now in restrictedConfig, we have the case where parent has two active children, plus we have this_tree on the left. We need to check if there are configurations where other_child is a dead node.
    ReticulationConfigSet combinedConfig2 = combineReticulationChoices(restrictedConfig, other_child_dead_settings);
    
    for (size_t i = 0; i < combinedConfig2.configs.size(); ++i) {
        res.configs.emplace_back(combinedConfig2.configs[i]);
    }

    simplifyReticulationChoices(res);    
    return res;
}

unsigned int processNodeImprovedTwoChildren(AnnotatedNetwork& ann_network, Node* node, Node* left_child, Node* right_child, const ReticulationConfigSet& extraRestrictions) {
    unsigned int num_trees_added = 0;
    size_t fake_clv_index = ann_network.network.nodes.size();
    
    pll_operation_t op_both = buildOperation(ann_network.network, node, left_child, right_child, ann_network.network.nodes.size(), ann_network.network.edges.size());

    ReticulationConfigSet restrictionsBothSet = getRestrictionsToTakeNeighbor(ann_network, node, left_child);
    restrictionsBothSet = combineReticulationChoices(restrictionsBothSet, getRestrictionsToTakeNeighbor(ann_network, node, right_child));
    if (!extraRestrictions.configs.empty()) {
        restrictionsBothSet = combineReticulationChoices(restrictionsBothSet, extraRestrictions);
    }

    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[node->clv_index];
    NodeDisplayedTreeData& displayed_trees_left_child = ann_network.pernode_displayed_tree_data[left_child->clv_index];
    NodeDisplayedTreeData& displayed_trees_right_child = ann_network.pernode_displayed_tree_data[right_child->clv_index];

    // left child and right child are not always about different reticulations... It can be that one reticulation affects both children.
    // It can even happen that there is a displayed tree for one child, that has no matching displaying tree on the other side (in terms of chosen reticulations). In this case, we have a dead node situation...
    
    // combine both children - here, both children are active
    for (size_t i = 0; i < displayed_trees_left_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& leftTree = displayed_trees_left_child.displayed_trees[i];
        if (!reticulationConfigsCompatible(leftTree.treeLoglData.reticulationChoices, restrictionsBothSet)) {
            continue;
        }
        for (size_t j = 0; j < displayed_trees_right_child.num_active_displayed_trees; ++j) {
            DisplayedTreeData& rightTree = displayed_trees_right_child.displayed_trees[j];
            if (!reticulationConfigsCompatible(rightTree.treeLoglData.reticulationChoices, restrictionsBothSet)) {
                continue;
            }
            if (reticulationConfigsCompatible(leftTree.treeLoglData.reticulationChoices, rightTree.treeLoglData.reticulationChoices)) {
                add_displayed_tree(ann_network, node->clv_index);
                num_trees_added++;
                
                DisplayedTreeData& newDisplayedTree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];
                for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
                    //skip remote partitions
                    if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
                        continue;
                    }
                    pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
                    double* parent_clv = newDisplayedTree.clv_vector[partition_idx];
                    unsigned int* parent_scaler = newDisplayedTree.scale_buffer[partition_idx];
                    double* left_clv = leftTree.clv_vector[partition_idx];
                    unsigned int* left_scaler = leftTree.scale_buffer[partition_idx];
                    double* right_clv = rightTree.clv_vector[partition_idx];
                    unsigned int* right_scaler = rightTree.scale_buffer[partition_idx];
                    if (node->clv_index == ann_network.network.root->clv_index) {
                        //std::cout << "I am thread " << ParallelContext::local_proc_id() << " in 2 children mode and I am updating a root clv for partition " << partition_idx << "\n";
                    }
                    //std::cout << "I am at node with clv index " << node->clv_index << "\n";
                    assert(leftTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], left_clv));
                    assert(rightTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], right_clv));
                    pll_update_partials_single(partition, &op_both, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
                    assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], parent_clv));
                }
                newDisplayedTree.treeLoglData.reticulationChoices = combineReticulationChoices(leftTree.treeLoglData.reticulationChoices, rightTree.treeLoglData.reticulationChoices);
                newDisplayedTree.treeLoglData.reticulationChoices = combineReticulationChoices(newDisplayedTree.treeLoglData.reticulationChoices, restrictionsBothSet);
                
                //newDisplayedTree.treeLoglData.childrenTaken = {left_child, right_child};
            }
        }
    }

    // only left child, not right child
    pll_operation_t op_left_only = buildOperation(ann_network.network, node, left_child, nullptr, ann_network.network.nodes.size(), ann_network.network.edges.size());
    ReticulationConfigSet right_child_dead_settings = deadNodeSettings(ann_network, displayed_trees_right_child, node, right_child);
    if (!extraRestrictions.configs.empty()) {
        right_child_dead_settings = combineReticulationChoices(right_child_dead_settings, extraRestrictions);
    }
    for (size_t i = 0; i < displayed_trees_left_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& leftTree = displayed_trees_left_child.displayed_trees[i];
        ReticulationConfigSet leftOnlyConfigs = getReticulationChoicesThisOnly(ann_network, leftTree.treeLoglData.reticulationChoices, right_child_dead_settings, node, left_child, right_child);
        if (!extraRestrictions.configs.empty()) {
            leftOnlyConfigs = combineReticulationChoices(leftOnlyConfigs, extraRestrictions);
        }

        if (!leftOnlyConfigs.configs.empty()) {
            add_displayed_tree(ann_network, node->clv_index);
            DisplayedTreeData& tree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];
            
            for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
                // skip remote partitions
                if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
                    continue;
                }
                pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
                double* parent_clv = tree.clv_vector[partition_idx];
                unsigned int* parent_scaler = tree.scale_buffer[partition_idx];
                double* left_clv = leftTree.clv_vector[partition_idx];
                unsigned int* left_scaler = leftTree.scale_buffer[partition_idx];
                double* right_clv = partition->clv[fake_clv_index];
                unsigned int* right_scaler = nullptr;
                assert(leftTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], left_clv));
                assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], right_clv));
                pll_update_partials_single(partition, &op_left_only, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
                assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], parent_clv));
            }
            tree.treeLoglData.reticulationChoices = leftOnlyConfigs;
            //tree.treeLoglData.childrenTaken = {left_child};
            num_trees_added++;
        }
    }

    // only right child, not left child
    pll_operation_t op_right_only = buildOperation(ann_network.network, node, right_child, nullptr, ann_network.network.nodes.size(), ann_network.network.edges.size());
    ReticulationConfigSet left_child_dead_settings = deadNodeSettings(ann_network, displayed_trees_left_child, node, left_child);
    if (!extraRestrictions.configs.empty()) {
        left_child_dead_settings = combineReticulationChoices(left_child_dead_settings, extraRestrictions);
    }
    for (size_t i = 0; i < displayed_trees_right_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& rightTree = displayed_trees_right_child.displayed_trees[i];
        ReticulationConfigSet rightOnlyConfigs = getReticulationChoicesThisOnly(ann_network, rightTree.treeLoglData.reticulationChoices, left_child_dead_settings, node, right_child, left_child);
        if (!extraRestrictions.configs.empty()) {
            rightOnlyConfigs = combineReticulationChoices(rightOnlyConfigs, extraRestrictions);
        }
        if (!rightOnlyConfigs.configs.empty()) {
            add_displayed_tree(ann_network, node->clv_index);
            DisplayedTreeData& tree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];

            for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
                // skip remote partitions
                if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
                    continue;
                }
                pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
                double* parent_clv = tree.clv_vector[partition_idx];
                unsigned int* parent_scaler = tree.scale_buffer[partition_idx];
                double* left_clv = rightTree.clv_vector[partition_idx];
                unsigned int* left_scaler = rightTree.scale_buffer[partition_idx];
                double* right_clv = partition->clv[fake_clv_index];
                unsigned int* right_scaler = nullptr;
                assert(rightTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], left_clv));
                assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], right_clv));
                pll_update_partials_single(partition, &op_right_only, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
                assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], parent_clv));
            }
            tree.treeLoglData.reticulationChoices = rightOnlyConfigs;
            //tree.treeLoglData.childrenTaken = {right_child};
            num_trees_added++;
        }
    }

    return num_trees_added;
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

void processNodeImproved(AnnotatedNetwork& ann_network, int incremental, Node* node, std::vector<Node*>& children, const ReticulationConfigSet& extraRestrictions, bool append = false) {
    if (node->clv_index < ann_network.network.num_tips()) {
        //assert(ann_network.fake_treeinfo->clv_valid[partition_idx][node->clv_index]);
        return;
    }
    if (incremental && allClvsValid(ann_network.fake_treeinfo, node->clv_index)) {
        return;
    }

    /*std::cout << "processNodeImproved called for node " << node->clv_index << " with children: ";
    for (size_t i = 0; i < children.size(); ++i) {
        std::cout << children[i]->clv_index;
        if (i + 1 < children.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";*/

    if (!append) {
        ann_network.pernode_displayed_tree_data[node->clv_index].num_active_displayed_trees = 0;
    }

    if (children.size() == 0) {
        //std::cout << exportDebugInfo(ann_network) << "\n";
        //std::cout << "Node " << node->clv_index << " has no children!!! It is a dead node! \n";
        //std::cout << "extra restrictions:\n";
        //printReticulationChoices(extraRestrictions);
        return;
    }

    if (children.size() == 1) {
        processNodeImprovedSingleChild(ann_network, node, children[0], extraRestrictions);
    } else {
        assert(children.size() == 2);
        Node* left_child = children[0];
        Node* right_child = children[1];
        processNodeImprovedTwoChildren(ann_network, node, left_child, right_child, extraRestrictions);
    }

    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        if (ann_network.fake_treeinfo->partitions[p]) {
            ann_network.fake_treeinfo->clv_valid[p][node->clv_index] = 1;
        }
    }
    
    if (ann_network.pernode_displayed_tree_data[node->clv_index].num_active_displayed_trees > (1 << ann_network.network.num_reticulations())) {
        std::cout << "Too many displayed trees stored at node " << node->clv_index << "\n";
    }
    assert(ann_network.pernode_displayed_tree_data[node->clv_index].num_active_displayed_trees <= (1 << ann_network.network.num_reticulations()));

    /*std::cout << "Node " << node->clv_index << " has been processed, displayed trees at the node are:\n";
    for (size_t i = 0; i < displayed_trees.num_active_displayed_trees; ++i) {
        printReticulationChoices(displayed_trees.displayed_trees[i].reticulationChoices);
    }*/
}

void processPartitionsImproved(AnnotatedNetwork& ann_network, int incremental) {
    //std::cout << "\nNEW PROCESS PARTITION_IMPROVED!!!\n";
    std::vector<bool> seen(ann_network.network.num_nodes(), false);
    
    for (size_t i = 0; i < ann_network.travbuffer.size(); ++i) {
        Node* actNode = ann_network.travbuffer[i];
        std::vector<Node*> children = getChildren(ann_network.network, actNode);
        if (children.size() > 2) {
            std::cout << exportDebugInfo(ann_network) << "\n";
            std::cout << "Node " << actNode->clv_index << " has too many children! The children are: \n";
            for (size_t j = 0; j < children.size(); ++j) {
                std::cout << children[j]->clv_index << "\n";
            }
        }
        assert(children.size() <= 2);
        processNodeImproved(ann_network, incremental, actNode, children, {});
    }

    for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees; ++i) {
        DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[i];
        computeDisplayedTreeLoglikelihood(ann_network, tree, ann_network.network.root);
    }

    /*std::cout << "\nloglikelihood for trees at each node, for partition " << partition_idx << ":\n";
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        std::cout << "node " << i << "\n";
        for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[partition_idx][i].num_active_displayed_trees; ++j) {
            std::cout << ann_network.pernode_displayed_tree_data[partition_idx][i].displayed_trees[j].tree_logl << "\n";
        }
    }*/
}

bool reuseOldDisplayedTreesCheck(AnnotatedNetwork& ann_network, int incremental) {
    if (!incremental) {
        return false;
    }
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool all_clvs_valid = true;
    for (size_t i = 0; i < fake_treeinfo.partition_count; ++i) {
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[i]) {
            continue;
        }
        all_clvs_valid &= fake_treeinfo.clv_valid[i][ann_network.network.root->clv_index];
    }
    return all_clvs_valid;
}

DisplayedTreeData& getMatchingDisplayedTreeAtNode(AnnotatedNetwork& ann_network, unsigned int partition_idx, unsigned int node_clv_index, const ReticulationConfigSet& queryChoices) {
    for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[node_clv_index].num_active_displayed_trees; ++i) {
        if (reticulationConfigsCompatible(queryChoices, ann_network.pernode_displayed_tree_data[node_clv_index].displayed_trees[i].treeLoglData.reticulationChoices)) {
            return ann_network.pernode_displayed_tree_data[node_clv_index].displayed_trees[i];
        }
    }
    throw std::runtime_error("No compatible displayed tree data found");
}

const TreeLoglData& getMatchingOldTree(AnnotatedNetwork& ann_network, const std::vector<DisplayedTreeData>& oldTrees, const ReticulationConfigSet& queryChoices) {
    for (size_t i = 0; i < oldTrees.size(); ++i) {
        if (reticulationConfigsCompatible(queryChoices, oldTrees[i].treeLoglData.reticulationChoices)) {
            return oldTrees[i].treeLoglData;
        }
    }
    std::cout << "query was:\n";
    printReticulationChoices(queryChoices);
    throw std::runtime_error("No compatible old tree data found");
}

double evaluateTreesPartition(AnnotatedNetwork& ann_network, size_t partition_idx, std::vector<TreeLoglData>& treeLoglData) {
    //ann_network.fake_treeinfo->active_partition = partition_idx;
    size_t n_trees = treeLoglData.size();

    if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
        mpfr::mpreal partition_lh = 0.0;
        for (size_t tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
            TreeLoglData& tree = treeLoglData[tree_idx];
            assert(tree.tree_logl_valid);
            assert(tree.tree_logprob_valid);
            assert(tree.tree_partition_logl[partition_idx] != 0.0);
            assert(tree.tree_partition_logl[partition_idx] != -std::numeric_limits<double>::infinity());
            assert(tree.tree_partition_logl[partition_idx] < 0.0);
            /*if (ann_network.network.num_reticulations() == 1) {
                std::cout << "tree " << tree_idx << " partition " << partition_idx << " logl: " << tree.tree_logl << "\n";
            }*/
            if (tree.tree_logprob != std::numeric_limits<double>::infinity()) {
                partition_lh += mpfr::exp(tree.tree_logprob) * mpfr::exp(tree.tree_partition_logl[partition_idx]);
            }
        }
        double partition_logl = mpfr::log(partition_lh).toDouble();
        /*if (ann_network.network.num_reticulations() == 1) {
            std::cout << "partition " << partition_idx << " logl: " << partition_logl << "\n";
        }*/
        ann_network.fake_treeinfo->partition_loglh[partition_idx] = partition_logl;
        return partition_logl;
    } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
        double partition_logl = -std::numeric_limits<double>::infinity();
        for (size_t tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
            TreeLoglData& tree = treeLoglData[tree_idx];
            if (!tree.tree_logl_valid) {
                throw std::runtime_error("invalid tree logl");
            }
            assert(tree.tree_logl_valid);
            assert(tree.tree_logprob_valid);
            assert(tree.tree_partition_logl[partition_idx] != 0.0);
            assert(tree.tree_partition_logl[partition_idx] != -std::numeric_limits<double>::infinity());
            assert(tree.tree_partition_logl[partition_idx] < 0.0);
            //std::cout << "tree " << tree_idx << " logl: " << tree.tree_logl << "\n";
            //std::cout << "tree " << tree_idx << " logprob: " << tree.tree_logprob << "\n";
            if (tree.tree_logprob != std::numeric_limits<double>::infinity()) {
                //std::cout << "tree " << tree.tree_idx << " prob: " << mpfr::exp(tree.tree_logprob) << "\n";
                partition_logl = std::max(partition_logl, tree.tree_logprob + tree.tree_partition_logl[partition_idx]);
            }
        }
        ann_network.fake_treeinfo->partition_loglh[partition_idx] = partition_logl;
        //std::cout << "partiion " << partition_idx << " logl: " << partition_logl << "\n";
        return partition_logl;
    }
}

double evaluateTreesPartition(AnnotatedNetwork& ann_network, size_t partition_idx, NodeDisplayedTreeData& nodeDisplayedTreeData) {
    //ann_network.fake_treeinfo->active_partition = partition_idx;
    std::vector<DisplayedTreeData>& displayed_root_trees = nodeDisplayedTreeData.displayed_trees;
    size_t n_trees = nodeDisplayedTreeData.num_active_displayed_trees;

    std::vector<TreeLoglData> treeLoglData;
    for (size_t i = 0; i < n_trees; ++i) {
        treeLoglData.emplace_back(displayed_root_trees[i].treeLoglData);
    }

    return evaluateTreesPartition(ann_network, partition_idx, treeLoglData);
}

double evaluateTrees(AnnotatedNetwork &ann_network, Node* virtual_root) {
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    mpfr::mpreal network_logl = 0.0;

    for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
        double partition_logl = evaluateTreesPartition(ann_network, partition_idx, ann_network.pernode_displayed_tree_data[virtual_root->clv_index]);
        network_logl += partition_logl;
    }

    fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    //std::cout << "network logl: " << network_logl.toDouble() << "\n";
    if (network_logl.toDouble() == -std::numeric_limits<double>::infinity()) {
        throw std::runtime_error("Invalid network likelihood: negative infinity \n");
    }
    ann_network.cached_logl = network_logl.toDouble();
    ann_network.cached_logl_valid = true;
    return ann_network.cached_logl;
}

bool isActiveBranch(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, unsigned int pmatrix_index) {
    Node* edge_source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* edge_target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);

    ReticulationConfigSet restrictions = getRestrictionsToTakeNeighbor(ann_network, edge_source, edge_target);
    return reticulationConfigsCompatible(restrictions, reticulationChoices);
}

std::vector<Node*> getPathToVirtualRoot(AnnotatedNetwork& ann_network, Node* from, Node* virtual_root, const std::vector<Node*> parent) {
    assert(from);
    assert(virtual_root);
    std::vector<Node*> res;
    Node* act_node = from;
    while (act_node != virtual_root){
        res.emplace_back(act_node);
        act_node = parent[act_node->clv_index];
    }
    res.emplace_back(virtual_root);
    return res;
}

std::vector<Node*> getParentPointers(AnnotatedNetwork& ann_network, const std::vector<ReticulationState>& reticulationChoices, Node* virtual_root) {
    assert(virtual_root);
    setReticulationParents(ann_network.network, reticulationChoices);
    std::vector<Node*> parent(ann_network.network.num_nodes(), nullptr);
    parent[virtual_root->clv_index] = virtual_root;
    std::queue<Node*> q;
    
    q.emplace(virtual_root);
    while (!q.empty()) {
        Node* actNode = q.front();
        q.pop();
        std::vector<Node*> neighbors = getActiveNeighbors(ann_network.network, actNode);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            Node* neigh = neighbors[i];
            if (!parent[neigh->clv_index]) { // neigh was not already processed
                q.emplace(neigh);
                parent[neigh->clv_index] = actNode;
            }
        }
    }
    parent[virtual_root->clv_index] = nullptr;
    return parent;
}

std::vector<Node*> getParentPointers(AnnotatedNetwork& ann_network, Node* virtual_root) {
    assert(virtual_root);
    std::vector<Node*> parent(ann_network.network.num_nodes(), nullptr);
    parent[virtual_root->clv_index] = virtual_root;
    std::queue<Node*> q;
    
    q.emplace(virtual_root);
    while (!q.empty()) {
        Node* actNode = q.front();
        q.pop();
        std::vector<Node*> neighbors = getActiveNeighbors(ann_network.network, actNode);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            Node* neigh = neighbors[i];
            if (!parent[neigh->clv_index]) { // neigh was not already processed
                q.emplace(neigh);
                parent[neigh->clv_index] = actNode;
            }
        }
    }
    parent[virtual_root->clv_index] = nullptr;
    return parent;
}

struct PathToVirtualRoot {
    ReticulationConfigSet reticulationChoices;
    std::vector<Node*> path;
    std::vector<std::vector<Node*>> children;

    PathToVirtualRoot(size_t max_reticulations) : reticulationChoices(max_reticulations) {};
};

void printPathToVirtualRoot(const PathToVirtualRoot& pathToVirtualRoot) {
    std::cout << "Path has reticulation choices:\n";
    printReticulationChoices(pathToVirtualRoot.reticulationChoices);
    for (size_t i = 0; i < pathToVirtualRoot.path.size(); ++i) {
        std::cout << "Node " << pathToVirtualRoot.path[i]->clv_index << " has children: ";
        for (size_t j = 0; j < pathToVirtualRoot.children[i].size(); ++j) {
            std::cout << pathToVirtualRoot.children[i][j]->clv_index;
            if (j + 1 < pathToVirtualRoot.children[i].size()) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    }
}

std::vector<Node*> getCurrentChildren(AnnotatedNetwork& ann_network, Node* node, Node* parent, const ReticulationConfigSet& restrictions) {
    assert(restrictions.configs.size() == 1);
    std::vector<Node*> children = getChildrenIgnoreDirections(ann_network.network, node, parent);
    std::vector<Node*> res;
    for (size_t i = 0; i < children.size(); ++i) {
        if (reticulationConfigsCompatible(restrictions, getRestrictionsToTakeNeighbor(ann_network, node, children[i]))) {
            res.emplace_back(children[i]);
        }
    }
    if (res.size() > 2) {
        std::cout << "Node: " << node->clv_index << "\n";
        if (parent) {
            std::cout << "Parent: " << parent->clv_index << "\n";
        } else {
            std::cout << "Parent: NULL" << "\n";
        }

        std::cout << exportDebugInfo(ann_network) << "\n";
    }
    assert(res.size() <= 2);
    return res;
}

std::vector<PathToVirtualRoot> getPathsToVirtualRoot(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back) {
    std::vector<PathToVirtualRoot> res;
    // naive version here: Go over all displayed trees, compute pathToVirtualRoot for each of them, and then later on kick out duplicate paths...

    /*if (ann_network.network.num_reticulations() == 1) {
        std::cout << "\nold virtual root: " << old_virtual_root->clv_index << "\n";
        std::cout << "new virtual root: " << new_virtual_root->clv_index << "\n";
        std::cout << "new virtual root back: " << new_virtual_root_back->clv_index << "\n";
    }*/

    NodeDisplayedTreeData& oldDisplayedTrees = ann_network.pernode_displayed_tree_data[old_virtual_root->clv_index];
    for (size_t i = 0; i < oldDisplayedTrees.num_active_displayed_trees; ++i) {
        PathToVirtualRoot ptvr(ann_network.options.max_reticulations);
        for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
            setReticulationParents(ann_network.network, oldDisplayedTrees.displayed_trees[i].treeLoglData.reticulationChoices.configs[0]);
        }
        std::vector<Node*> parent = getParentPointers(ann_network, new_virtual_root);
        std::vector<Node*> path = getPathToVirtualRoot(ann_network, old_virtual_root, new_virtual_root, parent);
        
        std::vector<ReticulationState> dont_care_vector(ann_network.options.max_reticulations, ReticulationState::DONT_CARE);
        ReticulationConfigSet restrictionsSet(ann_network.options.max_reticulations);
        restrictionsSet.configs.emplace_back(dont_care_vector);

        for (size_t j = 0; j < path.size() - 1; ++j) {
            restrictionsSet = combineReticulationChoices(restrictionsSet, getRestrictionsToTakeNeighbor(ann_network, path[j], path[j+1]));
        }
        assert(restrictionsSet.configs.size() == 1);
        ptvr.reticulationChoices = restrictionsSet;
        ptvr.path = path;

        /*std::cout << "parent pointers for reticulation choices:\n";
        printReticulationChoices(restrictionsSet);
        for (size_t j = 0; j < path.size(); ++j) {
            if (parent[path[j]->clv_index]) {
                std::cout << path[j]->clv_index << " has parent " << parent[path[j]->clv_index]->clv_index << "\n";
            } else {
                std::cout << path[j]->clv_index << " has NO parent\n";
            }
        }
        std::cout << "\n";*/

        assert(ptvr.path[0] == old_virtual_root);
        for (size_t j = 0; j < path.size() - 1; ++j) {
            if (path[j] == new_virtual_root_back) {
                ptvr.children.emplace_back(getCurrentChildren(ann_network, path[j], new_virtual_root, restrictionsSet)); // Not sure if this special case is needed
            } else {
                ptvr.children.emplace_back(getCurrentChildren(ann_network, path[j], parent[path[j]->clv_index], restrictionsSet));
            }
        }
        // Special case at new virtual root, there we don't want to include new_virtual_root_back...
        assert(path[path.size() - 1] == new_virtual_root);
        ptvr.children.emplace_back(getCurrentChildren(ann_network, new_virtual_root, new_virtual_root_back, restrictionsSet));

        res.emplace_back(ptvr);
    }

    // Kick out the duplicate paths
    bool foundDuplicate = true;
    while (foundDuplicate) {
        foundDuplicate = false;
        for (size_t i = 0; i < res.size() - 1; ++i) {
            for (size_t j = i + 1; j < res.size(); ++j) {
                if (res[i].path == res[j].path) {
                    foundDuplicate = true;
                    std::swap(res[j], res[res.size() - 1]);
                    res.pop_back();
                    break;
                }
            }
            if (foundDuplicate) {
                break;
            }
        }
    }

    /*if (ann_network.network.num_reticulations() == 1 && (new_virtual_root_back->clv_index == 15)) {
        std::cout << "paths from old virtual root " << old_virtual_root->clv_index << " to virtual root " << new_virtual_root->clv_index << ":\n";
        for (size_t i = 0; i < res.size(); ++i) {
            printPathToVirtualRoot(res[i]);
        }
    }*/

    return res;
}

struct NodeSaveInformation {
    std::vector<std::unordered_set<size_t> > pathNodesToRestore;
    std::unordered_set<size_t> nodesInDanger;
};

NodeSaveInformation computeNodeSaveInformation(const std::vector<PathToVirtualRoot>& paths) {
    NodeSaveInformation nodeSaveInfo;
    std::vector<std::unordered_set<size_t> >& pathNodesToRestore = nodeSaveInfo.pathNodesToRestore;
    std::unordered_set<size_t>& nodesInDanger = nodeSaveInfo.nodesInDanger;
    
    pathNodesToRestore.resize(paths.size());
    for (size_t p = 1; p < paths.size(); ++p) {
        std::unordered_set<size_t> nodesInPath;
        for (size_t i = 0; i < paths[p].path.size(); ++i) {
            nodesInPath.emplace(paths[p].path[i]->clv_index);
            for (size_t j = 0; j < paths[p].children[i].size(); ++j) {
                pathNodesToRestore[p].emplace(paths[p].children[i][j]->clv_index);
            }
        }
        for (size_t nodeInPath : nodesInPath) {
            pathNodesToRestore[p].erase(nodeInPath);
        }

        // check if the node occurs at an earlier path. If so, it really needs to be restored, as that earlier path would have overwritten it.
        std::unordered_set<size_t> deleteAgain;

        for (size_t maybeSaveMe : pathNodesToRestore[p]) {
            bool saveMe = false;
            for (size_t q = 0; q < p; ++q) {
                for (size_t i = 0; i < paths[q].path.size(); ++i) {
                    if (paths[q].path[i]->clv_index == maybeSaveMe) {
                        saveMe = true;
                        break;
                    }
                }
                if (saveMe) {
                    break;
                }
            }
            if (!saveMe) {
                deleteAgain.emplace(maybeSaveMe);
            }
        }

        for (size_t deleteMeAgain : deleteAgain) {
            pathNodesToRestore[p].erase(deleteMeAgain);
        }
    }

    for (size_t p = 0; p < paths.size(); ++p) {
        for (size_t nodeToSave : pathNodesToRestore[p]) {
            nodesInDanger.emplace(nodeToSave);
        }
    }

    return nodeSaveInfo;
}

void updateCLVsVirtualRerootTrees(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back) {
    assert(old_virtual_root);
    assert(new_virtual_root);

    // 1.) for all paths from retNode to new_virtual_root:
    //     1.1) Collect the reticulation nodes encountered on the path, build exta restrictions storing the reticulation configurations used
    //     1.2) update CLVs on that path, using extra restrictions and append mode
    std::vector<PathToVirtualRoot> paths = getPathsToVirtualRoot(ann_network, old_virtual_root, new_virtual_root, new_virtual_root_back);

    // figure out which nodes to save from old CLVs, and which paths need them
    NodeSaveInformation nodeSaveInfo = computeNodeSaveInformation(paths);
    std::vector<NodeDisplayedTreeData> bufferedNodeInformations(ann_network.network.num_nodes());
    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        for (size_t nodeInDanger : nodeSaveInfo.nodesInDanger) {
            bufferedNodeInformations[nodeInDanger] = ann_network.pernode_displayed_tree_data[nodeInDanger];
        }
    }
    
    for (size_t p = 0; p < paths.size(); ++p) {
        /*std::cout << "PROCESSING PATH " << p << " ON PARTITION " << partition_idx << "\n";
        printPathToVirtualRoot(paths[p]);
        std::cout << "The path has the following restrictions: \n";
        printReticulationChoices(paths[p].reticulationChoices);*/

        // Restore required old NodeInformations for the path
        for (size_t nodeIndexToRestore : nodeSaveInfo.pathNodesToRestore[p]) {
            /*if (ann_network.network.num_reticulations() == 1) {
                std::cout << "restoring node info at " << nodeIndexToRestore << "\n";
            }*/
            ann_network.pernode_displayed_tree_data[nodeIndexToRestore] = bufferedNodeInformations[nodeIndexToRestore];
        }

        for (size_t i = 0; i < paths[p].path.size(); ++i) {
            bool appendMode = ((p > 0) && (paths[p].path[i] == new_virtual_root));
            assert((paths[p].path[i] != new_virtual_root) || ((paths[p].path[i] == new_virtual_root) && (i == paths[p].path.size() - 1)));
            processNodeImproved(ann_network, 0, paths[p].path[i], paths[p].children[i], paths[p].reticulationChoices, appendMode);
        }
    }
}

void printDisplayedTreesChoices(AnnotatedNetwork& ann_network, size_t partition_idx, Node* virtualRoot) {
    NodeDisplayedTreeData& nodeData = ann_network.pernode_displayed_tree_data[virtualRoot->clv_index];
    for (size_t i = 0; i < nodeData.num_active_displayed_trees; ++i) {
        printReticulationChoices(nodeData.displayed_trees[i].treeLoglData.reticulationChoices);
    }
}

struct TreeDerivatives {
    mpfr::mpreal lh_prime = 0.0;
    mpfr::mpreal lh_prime_prime = 0.0;
};

TreeDerivatives computeTreeDerivatives(double logl, double logl_prime, double logl_prime_prime) {
    TreeDerivatives res;
    mpfr::mpreal lh = mpfr::exp(logl);
    mpfr::mpreal lh_prime = lh * logl_prime;
    mpfr::mpreal lh_prime_prime = lh_prime * logl_prime + lh * logl_prime_prime;

    /*std::cout << "tree_logl: " << logl << "\n";
    std::cout << "tree_logl_prime: " << logl_prime << "\n";
    std::cout << "tree_logl_prime_prime: " << logl_prime_prime << "\n";

    std::cout << "tree_lh: " << lh << "\n";
    std::cout << "tree_lh_prime: " << lh_prime << "\n";
    std::cout << "tree_lh_prime_prime: " << lh_prime_prime << "\n";*/

    res.lh_prime = lh_prime;
    res.lh_prime_prime = lh_prime_prime;
    return res;
}

struct PartitionLhData {
    double logl_prime = 0.0;
    double logl_prime_prime = 0.0;
};

PartitionLhData computePartitionLhData(AnnotatedNetwork& ann_network, unsigned int partition_idx, const std::vector<SumtableInfo>& sumtables, const std::vector<DisplayedTreeData>& oldTrees, unsigned int pmatrix_index) {
    PartitionLhData res{0.0, 0.0};
    Node* source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);

    bool single_tree_mode = (sumtables.size() == 1);

    /*size_t n_trees_source = ann_network.pernode_displayed_tree_data[partition_idx][source->clv_index].num_active_displayed_trees;
    size_t n_trees_target = ann_network.pernode_displayed_tree_data[partition_idx][target->clv_index].num_active_displayed_trees;
    std::vector<DisplayedTreeData>& sourceTrees = ann_network.pernode_displayed_tree_data[partition_idx][source->clv_index].displayed_trees;
    std::vector<DisplayedTreeData>& targetTrees = ann_network.pernode_displayed_tree_data[partition_idx][target->clv_index].displayed_trees;
    std::vector<bool> source_tree_seen(n_trees_source, false);
    std::vector<bool> target_tree_seen(n_trees_target, false);*/

    // TODO: Get NetRAX to correctly work with scaled brlens...
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
        throw std::runtime_error("I believe this function currently does not work correctly with scaled branch lengths");
    }

    double s = 1.0;
    //double s = ann_network.fake_treeinfo->brlen_scalers ? ann_network.fake_treeinfo->brlen_scalers[partition_idx] : 1.;
    double p_brlen = s * ann_network.fake_treeinfo->branch_lengths[partition_idx][pmatrix_index];

    //mpfr::mpreal logl = 0.0;
    mpfr::mpreal lh_sum = 0.0;
    mpfr::mpreal lh_prime_sum = 0.0;
    mpfr::mpreal lh_prime_prime_sum = 0.0;

    double best_tree_logl_score = -std::numeric_limits<double>::infinity();
    double best_tree_logl_prime_score = -std::numeric_limits<double>::infinity();
    double best_tree_logl_prime_prime_score = -std::numeric_limits<double>::infinity();

    double branch_length = ann_network.fake_treeinfo->branch_lengths[partition_idx][pmatrix_index];
    double ** eigenvals = nullptr;
    double * prop_invar = nullptr;
    double * diagptable = nullptr;

    if (ann_network.fake_treeinfo->partitions[partition_idx]) {
        pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
        pll_compute_eigenvals_and_prop_invar(partition, ann_network.fake_treeinfo->param_indices[partition_idx], &eigenvals, &prop_invar);
        diagptable = pll_compute_diagptable(partition->states, partition->rate_cats, branch_length, prop_invar, partition->rates, eigenvals);
        free (eigenvals);
    }

    /*if (ann_network.network.num_reticulations() == 1) {
        std::cout << "\ncomputePartitionLoglData for partition " << partition_idx << ":\n";
        std::cout << "number of sumtables: " << sumtables.size() << "\n";
    }*/
    for (size_t i = 0; i < sumtables.size(); ++i) {
        //source_tree_seen[sumtables[i].left_tree_idx] = true;
        //target_tree_seen[sumtables[i].right_tree_idx] = true;

        double tree_logl = 0.0;
        double tree_logl_prime = 0.0;
        double tree_logl_prime_prime = 0.0;

        if (ann_network.fake_treeinfo->partitions[partition_idx]) {
            pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];
            pll_compute_loglikelihood_derivatives(partition, 
                                            source->scaler_index, 
                                            sumtables[i].left_tree->scale_buffer[partition_idx],
                                            target->scaler_index, 
                                            sumtables[i].right_tree->scale_buffer[partition_idx],
                                            p_brlen,
                                            ann_network.fake_treeinfo->param_indices[partition_idx],
                                            sumtables[i].sumtable,
                                            (single_tree_mode) ? nullptr : &tree_logl,
                                            &tree_logl_prime,
                                            &tree_logl_prime_prime,
                                            diagptable,
                                            prop_invar);
        }

        /* sum up values from all threads */
        if (ann_network.fake_treeinfo->parallel_reduce_cb)
        {
            if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED)
            {
                if (!single_tree_mode) {
                    ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context, &tree_logl, 
                                                ann_network.fake_treeinfo->partition_count, PLLMOD_COMMON_REDUCE_SUM);
                }
                ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context, &tree_logl_prime, 
                                            ann_network.fake_treeinfo->partition_count, PLLMOD_COMMON_REDUCE_SUM);
                ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context, &tree_logl_prime_prime,
                                            ann_network.fake_treeinfo->partition_count, PLLMOD_COMMON_REDUCE_SUM);
            }
            else
            {
                if (single_tree_mode) {
                    double d[2] = {tree_logl_prime, tree_logl_prime_prime};
                    ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context, d, 2, PLLMOD_COMMON_REDUCE_SUM);
                    tree_logl_prime = d[0];
                    tree_logl_prime_prime = d[1];
                } else {
                    double d[3] = {tree_logl, tree_logl_prime, tree_logl_prime_prime};
                    ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context, d, 3, PLLMOD_COMMON_REDUCE_SUM);
                    tree_logl = d[0];
                    tree_logl_prime = d[1];
                    tree_logl_prime_prime = d[2];
                }
            }
        }

        /*if (ann_network.network.num_reticulations() == 1) {
            std::cout << "  tree_logl: " << tree_logl << "\n";
            std::cout << "  tree_logl_prime: " << tree_logl_prime << "\n";
            std::cout << "  tree_logl_prime_prime: " << tree_logl_prime_prime << "\n";
        }*/
        //assert(tree_logl != 0.0);

        if (single_tree_mode) {
            pll_aligned_free (diagptable);
            free (prop_invar);
            res.logl_prime = tree_logl_prime;
            res.logl_prime_prime = tree_logl_prime_prime;
            return res;
        }

        if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
            TreeDerivatives treeDerivatives = computeTreeDerivatives(tree_logl, tree_logl_prime, tree_logl_prime_prime);
            lh_sum += mpfr::exp(tree_logl) * sumtables[i].tree_prob;
            lh_prime_sum += treeDerivatives.lh_prime * sumtables[i].tree_prob;
            lh_prime_prime_sum += treeDerivatives.lh_prime_prime * sumtables[i].tree_prob;
        } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
            if (tree_logl * sumtables[i].tree_prob > best_tree_logl_score) {
                best_tree_logl_score = tree_logl * sumtables[i].tree_prob;
                best_tree_logl_prime_score = tree_logl_prime;
                best_tree_logl_prime_prime_score = tree_logl_prime_prime;
            }
        }
    }

    pll_aligned_free (diagptable);
    free (prop_invar);

    if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
        res.logl_prime = (lh_prime_sum / lh_sum).toDouble();
        res.logl_prime_prime = ((lh_prime_prime_sum * lh_sum - lh_prime_sum * lh_prime_sum) / (lh_sum * lh_sum)).toDouble();
    } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
        res.logl_prime = best_tree_logl_prime_score;
        res.logl_prime_prime = best_tree_logl_prime_prime_score;
    }

    /*
    for (size_t i = 0; i < n_trees_source; ++i) {
        if (source_tree_seen[i]) continue;
        for (size_t j = 0; j < n_trees_target; ++j) {
            if (target_tree_seen[j]) continue;

            TreeLoglData combinedTreeData;
            combinedTreeData.reticulationChoices = combineReticulationChoices(sourceTrees[i].treeLoglData.reticulationChoices, targetTrees[j].treeLoglData.reticulationChoices);
            if (!isActiveBranch(ann_network, combinedTreeData.reticulationChoices, pmatrix_index)) {
                source_tree_seen[i] = true;
                target_tree_seen[j] = true;
                //std::cout << "inactive branch\n";
                const TreeLoglData& oldTree = getMatchingOldTree(ann_network, oldTrees[partition_idx], combinedTreeData.reticulationChoices);
                if (!oldTree.tree_logl_valid) {
                    std::cout << exportDebugInfo(ann_network) << "\n";
                    std::cout << "i: " << i << "\n";
                    std::cout << "j: " << j << "\n";
                    std::cout << "pmatrix_index: " << pmatrix_index << "\n";
                    std::cout << "source: " << source->clv_index << "\n";
                    std::cout << "target: " << target->clv_index << "\n";
                }
                assert(oldTree.tree_logl_valid);
                combinedTreeData.tree_logl = oldTree.tree_logl;
                assert(combinedTreeData.tree_logl != -std::numeric_limits<double>::infinity());
                assert(oldTree.tree_logprob_valid);
                combinedTreeData.tree_logprob = oldTree.tree_logprob;

                if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
                    logl += mpfr::exp(combinedTreeData.tree_logl) * mpfr::exp(combinedTreeData.tree_logprob);
                } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
                    logl += std::max(logl, mpfr::exp(combinedTreeData.tree_logl) * mpfr::exp(combinedTreeData.tree_logprob));
                }

            }
        }
    }

    for (size_t i = 0; i < n_trees_source; ++i) {
        if (!source_tree_seen[i]) {
            //std::cout << "unseen source tree\n";
            const TreeLoglData& oldTree = getMatchingOldTree(ann_network, oldTrees[partition_idx], sourceTrees[i].treeLoglData.reticulationChoices);
            assert(oldTree.tree_logl_valid);
            sourceTrees[i].treeLoglData.tree_logl = oldTree.tree_logl;
            assert(oldTree.tree_logprob_valid);
            sourceTrees[i].treeLoglData.tree_logprob = oldTree.tree_logprob;
            sourceTrees[i].treeLoglData.tree_logl_valid = true;
            sourceTrees[i].treeLoglData.tree_logprob_valid = true;
            assert(sourceTrees[i].treeLoglData.tree_logl_valid);
            assert(sourceTrees[i].treeLoglData.tree_logl != -std::numeric_limits<double>::infinity());

            if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
                logl += mpfr::exp(sourceTrees[i].treeLoglData.tree_logl) * mpfr::exp(sourceTrees[i].treeLoglData.tree_logprob);
            } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
                logl += std::max(logl, mpfr::exp(sourceTrees[i].treeLoglData.tree_logl) * mpfr::exp(sourceTrees[i].treeLoglData.tree_logprob));
            }
        }
    }

    for (size_t j = 0; j < n_trees_target; ++j) {
        if (!target_tree_seen[j]) {
            //std::cout << "unseen target tree\n";
            const TreeLoglData& oldTree = getMatchingOldTree(ann_network, oldTrees[partition_idx], targetTrees[j].treeLoglData.reticulationChoices);
            assert(oldTree.tree_logl_valid);
            targetTrees[j].treeLoglData.tree_logl = oldTree.tree_logl;
            assert(oldTree.tree_logprob_valid);
            targetTrees[j].treeLoglData.tree_logprob = oldTree.tree_logprob;
            targetTrees[j].treeLoglData.tree_logl_valid = true;
            targetTrees[j].treeLoglData.tree_logprob_valid = true;
            assert(targetTrees[j].treeLoglData.tree_logl_valid);
            assert(targetTrees[j].treeLoglData.tree_logl != -std::numeric_limits<double>::infinity());

            if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
                logl += mpfr::exp(targetTrees[j].treeLoglData.tree_logl) * mpfr::exp(targetTrees[j].treeLoglData.tree_logprob);
            } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
                logl += std::max(logl, mpfr::exp(targetTrees[j].treeLoglData.tree_logl) * mpfr::exp(targetTrees[j].treeLoglData.tree_logprob));
            }
        }
    }
    */

    //res.logl = mpfr::log(logl).toDouble();

    /*if (ann_network.network.num_reticulations() == 1) {
        std::cout << " partition_index: " << partition_idx << "\n";
        std::cout << " partition_logl: " << res.logl << "\n";
        std::cout << " partition_logl_prime: " << res.logl_prime << "\n";
        std::cout << " partition_logl_prime_prime: " << res.logl_prime_prime << "\n\n";
    }*/

    //res.lh_prime *= s;
    //res.lh_prime_prime *= s * s;
    return res;
}

LoglDerivatives computeLoglikelihoodDerivatives(AnnotatedNetwork& ann_network, const std::vector<std::vector<SumtableInfo> >& sumtables, const std::vector<DisplayedTreeData>& oldTrees, unsigned int pmatrix_index, bool incremental, bool update_pmatrices) {
    //setup_pmatrices(ann_network, incremental, update_pmatrices);
    //double network_logl = 0.0;
    double network_logl_prime = 0.0;
    double network_logl_prime_prime = 0.0;
    assert(sumtables.size() == ann_network.fake_treeinfo->partition_count);
    std::vector<double> partition_logls_prime(ann_network.fake_treeinfo->partition_count, 0.0);
    std::vector<double> partition_logls_prime_prime(ann_network.fake_treeinfo->partition_count, 0.0);

    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) { // here we need to go over all partitions, as the derivatives require exact tree loglikelihood
        PartitionLhData pdata = computePartitionLhData(ann_network, p, sumtables[p], oldTrees, pmatrix_index);

        //std::cout << " Network partition loglikelihood derivatives for partition " << p << ":\n";
        //std::cout << " partition_logl: " << pdata.logl << "\n";
        //std::cout << " partition_logl_prime: " << pdata.logl_prime << "\n";
        //std::cout << " partition_logl_prime_prime: " << pdata.logl_prime_prime << "\n";

        partition_logls_prime[p] = pdata.logl_prime;
        partition_logls_prime_prime[p] = pdata.logl_prime_prime;

        //network_logl += pdata.logl;
        network_logl_prime += pdata.logl_prime;
        network_logl_prime_prime += pdata.logl_prime_prime;
    }

    //std::cout << "Network loglikelihood derivatives:\n";
    //std::cout << "network_logl: " << network_logl << "\n";
    //std::cout << "network_logl_prime: " << network_logl_prime << "\n";
    //std::cout << "network_logl_prime_prime: " << network_logl_prime_prime << "\n";
    //std::cout << "network_logl_prime / network_logl_prime_prime: " << network_logl_prime / network_logl_prime_prime << "\n";
    //std::cout << "\n";
    return LoglDerivatives{network_logl_prime, network_logl_prime_prime, partition_logls_prime, partition_logls_prime_prime};
}

/*double computeLoglikelihoodFromSumtables(AnnotatedNetwork& ann_network, const std::vector<std::vector<SumtableInfo> >& sumtables, const std::vector<std::vector<TreeLoglData> >& oldTrees, unsigned int pmatrix_index, bool incremental, bool update_pmatrices) {
    setup_pmatrices(ann_network, incremental, update_pmatrices);
    mpfr::mpreal network_logl = 0.0;

    assert(sumtables.size() == ann_network.fake_treeinfo->partition_count);

    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        PartitionLhData pdata = computePartitionLhData(ann_network, p, sumtables[p], oldTrees, pmatrix_index);
        network_logl += pdata.logl;
    }
    return network_logl.toDouble();
}*/

SumtableInfo computeSumtable(AnnotatedNetwork& ann_network, size_t partition_idx, const ReticulationConfigSet& restrictions, DisplayedTreeData& left_tree, size_t left_clv_index, DisplayedTreeData& right_tree, size_t right_clv_index, size_t left_tree_idx, size_t right_tree_idx) {
    //skip remote partitions
    if (!ann_network.fake_treeinfo->partitions[partition_idx]) { // add an empty fake sumtable
        SumtableInfo sumtableInfo(0, 0, &left_tree, &right_tree, left_tree_idx, right_tree_idx);
        sumtableInfo.tree_prob = computeReticulationConfigProb(restrictions, ann_network.reticulation_probs);
        return sumtableInfo;
    }
    
    pll_partition_t * partition = ann_network.fake_treeinfo->partitions[partition_idx];
    size_t sumtableSize = (partition->sites + partition->states) * partition->rate_cats * partition->states_padded;
    SumtableInfo sumtableInfo(sumtableSize, partition->alignment, &left_tree, &right_tree, left_tree_idx, right_tree_idx);

    sumtableInfo.tree_prob = computeReticulationConfigProb(restrictions, ann_network.reticulation_probs);
    sumtableInfo.sumtable = (double*) pll_aligned_alloc(sumtableSize * sizeof(double), partition->alignment);
    if (!sumtableInfo.sumtable) {
        throw std::runtime_error("Error in allocating memory for sumtable");
    }
    pll_update_sumtable(partition, left_clv_index, left_tree.clv_vector[partition_idx], right_clv_index, right_tree.clv_vector[partition_idx], left_tree.scale_buffer[partition_idx], right_tree.scale_buffer[partition_idx], ann_network.fake_treeinfo->param_indices[partition_idx], sumtableInfo.sumtable);

    return sumtableInfo;
}

std::vector<std::vector<SumtableInfo> > computePartitionSumtables(AnnotatedNetwork& ann_network, unsigned int pmatrix_index) {
    std::vector<std::vector<SumtableInfo> > res(ann_network.fake_treeinfo->partition_count);
    Node* source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);

    size_t n_trees_source = ann_network.pernode_displayed_tree_data[source->clv_index].num_active_displayed_trees;
    size_t n_trees_target = ann_network.pernode_displayed_tree_data[target->clv_index].num_active_displayed_trees;
    std::vector<DisplayedTreeData>& sourceTrees = ann_network.pernode_displayed_tree_data[source->clv_index].displayed_trees;
    std::vector<DisplayedTreeData>& targetTrees = ann_network.pernode_displayed_tree_data[target->clv_index].displayed_trees;

    for (size_t i = 0; i < n_trees_source; ++i) {
        for (size_t j = 0; j < n_trees_target; ++j) {
            if (!reticulationConfigsCompatible(sourceTrees[i].treeLoglData.reticulationChoices, targetTrees[j].treeLoglData.reticulationChoices)) {
                continue;
            }

            ReticulationConfigSet restrictions = combineReticulationChoices(sourceTrees[i].treeLoglData.reticulationChoices, targetTrees[j].treeLoglData.reticulationChoices);
            if (isActiveBranch(ann_network, restrictions, pmatrix_index)) {
                for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {         // here we need all partitions, as we require the sumtable info metadata
                    res[p].emplace_back(std::move(computeSumtable(ann_network, p, restrictions, sourceTrees[i], source->clv_index, targetTrees[j], target->clv_index, i, j)));
                }
            }
        }
    }

    // TODO: This is just for comining left and right tree. Sometimes, we need a single tree...
    return res;
}

double computeLoglikelihoodBrlenOpt(AnnotatedNetwork &ann_network, const std::vector<DisplayedTreeData>& oldTrees, unsigned int pmatrix_index, int incremental, int update_pmatrices) {
    if (ann_network.cached_logl_valid) {
        return ann_network.cached_logl;
    }
    ann_network.fake_treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    
    Node* source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    assert(reuseOldDisplayedTreesCheck(ann_network, incremental)); // TODO: Doesn't this need the virtual_root pointer, too?
    setup_pmatrices(ann_network, incremental, update_pmatrices);

    /*if (ann_network.network.num_reticulations() == 1 && pmatrix_index == 15) {
        for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
            std::cout << "Displayed trees stored at node " << i << ":\n";
            size_t n_trees = ann_network.pernode_displayed_tree_data[0][i].num_active_displayed_trees;
            for (size_t j = 0; j < n_trees; ++j) {
                DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[0][i].displayed_trees[j];
                printReticulationChoices(tree.reticulationChoices);
            }
        }
    }*/

    //Instead of going over the-source-trees-only for final loglh evaluation, we need to go over all pairs of trees, one in source node and one in target node.

    double network_logl = 0;

    
    std::vector<TreeLoglData> combinedTrees; // TODO
    
    size_t n_trees_source = ann_network.pernode_displayed_tree_data[source->clv_index].num_active_displayed_trees;
    size_t n_trees_target = ann_network.pernode_displayed_tree_data[target->clv_index].num_active_displayed_trees;
    std::vector<DisplayedTreeData>& sourceTrees = ann_network.pernode_displayed_tree_data[source->clv_index].displayed_trees;
    std::vector<DisplayedTreeData>& targetTrees = ann_network.pernode_displayed_tree_data[target->clv_index].displayed_trees;

    std::vector<bool> source_tree_seen(n_trees_source, false);
    std::vector<bool> target_tree_seen(n_trees_target, false);

    for (size_t i = 0; i < n_trees_source; ++i) {
        for (size_t j = 0; j < n_trees_target; ++j) {
            if (!reticulationConfigsCompatible(sourceTrees[i].treeLoglData.reticulationChoices, targetTrees[j].treeLoglData.reticulationChoices)) {
                continue;
            }
            source_tree_seen[i] = true;
            target_tree_seen[j] = true;

            TreeLoglData combinedTreeData(ann_network.fake_treeinfo->partition_count, ann_network.options.max_reticulations);
            combinedTreeData.reticulationChoices = combineReticulationChoices(sourceTrees[i].treeLoglData.reticulationChoices, targetTrees[j].treeLoglData.reticulationChoices);

            /*if (ann_network.network.num_reticulations() == 1) {
                std::cout << "Reticulation choices source tree " << i << ":\n";
                printReticulationChoices(sourceTrees[i].treeLoglData.reticulationChoices);
                std::cout << "Reticulation choices target tree " << j << ":\n";
                printReticulationChoices(targetTrees[j].treeLoglData.reticulationChoices);
                std::cout << "Reticulation choices combined " << j << ":\n";
                printReticulationChoices(combinedTreeData.reticulationChoices);
            }*/

            if (isActiveBranch(ann_network, combinedTreeData.reticulationChoices, pmatrix_index)) {
                //std::cout << std::setprecision(70);
                //std::cout << "active branch case, combining " << source->clv_index << " and " << target->clv_index << " for branch " << pmatrix_index << " with length " << ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] << "\n";
                //std::cout << "source CLV vector at " << source->clv_index << "\n";
                //printClv(*ann_network.fake_treeinfo, source->clv_index, sourceTrees[i].clv_vector, p);
                //std::cout << "target CLV vector at " << target->clv_index << "\n";
                //printClv(*ann_network.fake_treeinfo, target->clv_index, targetTrees[j].clv_vector, p);

                for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                    combinedTreeData.tree_partition_logl[p] = 0.0;
                    // skip remote partitions
                    if (!ann_network.fake_treeinfo->partitions[p]) {
                        continue;
                    }

                    pll_partition_t* partition = ann_network.fake_treeinfo->partitions[p];
                    std::vector<double> persite_logl(ann_network.fake_treeinfo->partitions[p]->sites);

                    assert(sourceTrees[i].isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[p], sourceTrees[i].clv_vector[p]));
                    assert(targetTrees[j].isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[p], targetTrees[j].clv_vector[p]));
                    combinedTreeData.tree_partition_logl[p] = pll_compute_edge_loglikelihood(partition, source->clv_index, sourceTrees[i].clv_vector[p], sourceTrees[i].scale_buffer[p], 
                                                                target->clv_index, targetTrees[j].clv_vector[p], targetTrees[j].scale_buffer[p], 
                                                                pmatrix_index, ann_network.fake_treeinfo->param_indices[p], persite_logl.data());
                    if (combinedTreeData.tree_partition_logl[p] == -std::numeric_limits<double>::infinity()) {
                        std::cout << exportDebugInfo(ann_network) << "\n";
                        std::cout << "i: " << i << "\n";
                        std::cout << "j: " << j << "\n";
                        std::cout << "pmatrix_index: " << pmatrix_index << "\n";
                        std::cout << "source: " << source->clv_index << "\n";
                        std::cout << "target: " << target->clv_index << "\n";
                    }
                    assert(combinedTreeData.tree_partition_logl[p] != -std::numeric_limits<double>::infinity());
                    assert(combinedTreeData.tree_partition_logl[p] < 0.0);
                }

                /* sum up likelihood from all threads */
                if (ann_network.fake_treeinfo->parallel_reduce_cb)
                {
                    ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context,
                                                combinedTreeData.tree_partition_logl.data(),
                                                ann_network.fake_treeinfo->partition_count,
                                                PLLMOD_COMMON_REDUCE_SUM);

                    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                        if (combinedTreeData.tree_partition_logl[p] == 0.0) {
                            throw std::runtime_error("bad partition logl");
                        }
                    }
                }

                combinedTreeData.tree_logprob = computeReticulationConfigLogProb(combinedTreeData.reticulationChoices, ann_network.reticulation_probs);

            } else {
                //std::cout << "inactive branch\n";
                const TreeLoglData& oldTree = getMatchingOldTree(ann_network, oldTrees, combinedTreeData.reticulationChoices);
                if (!oldTree.tree_logl_valid) {
                    std::cout << exportDebugInfo(ann_network) << "\n";
                    std::cout << "i: " << i << "\n";
                    std::cout << "j: " << j << "\n";
                    std::cout << "pmatrix_index: " << pmatrix_index << "\n";
                    std::cout << "source: " << source->clv_index << "\n";
                    std::cout << "target: " << target->clv_index << "\n";
                }
                assert(oldTree.tree_logl_valid);
                for (size_t p = 0; p < ann_network.network.num_reticulations(); ++p) {
                    assert(oldTree.tree_partition_logl[p] <= 0.0);
                }
                combinedTreeData.tree_partition_logl = oldTree.tree_partition_logl;
                assert(oldTree.tree_logprob_valid);
                combinedTreeData.tree_logprob = oldTree.tree_logprob;
            }
            combinedTreeData.tree_logl_valid = true;
            combinedTreeData.tree_logprob_valid = true;
            combinedTrees.emplace_back(combinedTreeData);
        }
    }

    for (size_t i = 0; i < n_trees_source; ++i) {
        if (!source_tree_seen[i]) {
            //std::cout << "unseen source tree\n";
            const TreeLoglData& oldTree = getMatchingOldTree(ann_network, oldTrees, sourceTrees[i].treeLoglData.reticulationChoices);
            assert(oldTree.tree_logl_valid);
            sourceTrees[i].treeLoglData.tree_partition_logl = oldTree.tree_partition_logl;
            assert(oldTree.tree_logprob_valid);
            for (size_t p = 0; p < ann_network.network.num_reticulations(); ++p) {
                assert(oldTree.tree_partition_logl[p] <= 0.0);
            }
            sourceTrees[i].treeLoglData.tree_logprob = oldTree.tree_logprob;
            sourceTrees[i].treeLoglData.tree_logl_valid = true;
            sourceTrees[i].treeLoglData.tree_logprob_valid = true;
            assert(sourceTrees[i].treeLoglData.tree_logl_valid);
            combinedTrees.emplace_back(sourceTrees[i].treeLoglData);
        }
    }

    for (size_t j = 0; j < n_trees_target; ++j) {
        if (!target_tree_seen[j]) {
            //std::cout << "unseen target tree\n";
            const TreeLoglData& oldTree = getMatchingOldTree(ann_network, oldTrees, targetTrees[j].treeLoglData.reticulationChoices);
            assert(oldTree.tree_logl_valid);
            targetTrees[j].treeLoglData.tree_partition_logl = oldTree.tree_partition_logl;
            assert(oldTree.tree_logprob_valid);
            for (size_t p = 0; p < ann_network.network.num_reticulations(); ++p) {
                assert(oldTree.tree_partition_logl[p] <= 0.0);
            }
            targetTrees[j].treeLoglData.tree_logprob = oldTree.tree_logprob;
            targetTrees[j].treeLoglData.tree_logl_valid = true;
            targetTrees[j].treeLoglData.tree_logprob_valid = true;
            assert(targetTrees[j].treeLoglData.tree_logl_valid);
            combinedTrees.emplace_back(targetTrees[j].treeLoglData);
        }
    }

    for (size_t c = 0; c < combinedTrees.size(); ++c) {
        assert(combinedTrees[c].tree_logl_valid);
    }

    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        network_logl += evaluateTreesPartition(ann_network, p, combinedTrees);
    }

    // TODO: Remove me again, this is just for debug
    //std::cout << "n_trees: " << ann_network.pernode_displayed_tree_data[0][source->clv_index].num_active_displayed_trees << "\n";
    //std::cout << "Displayed trees to evaluate:\n";
    //printDisplayedTreesChoices(ann_network, source);
    /*
    std::cout << "computeLoglikelihoodBrlenOpt has been called\n";

    double new_logl_result = network_logl;
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        invalidateHigherCLVs(ann_network, getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]), p, true);
    }
    double old_logl_result = computeLoglikelihood(ann_network, 0, 1);
    if (new_logl_result != old_logl_result && fabs(new_logl_result - old_logl_result) >= 1E-3) {
        std::cout << "new_logl_result: " << new_logl_result << "\n";
        std::cout << "old_logl_result: " << old_logl_result << "\n";
        std::cout << exportDebugInfo(ann_network) << "\n";
    }
    assert(fabs(new_logl_result - old_logl_result) < 1E-3);
    Node* new_virtual_root = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* new_virtual_root_back = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    updateCLVsVirtualRerootTrees(ann_network, ann_network.network.root, new_virtual_root, new_virtual_root_back);
    */

    ann_network.cached_logl = network_logl;
    ann_network.cached_logl_valid = true;

    //std::cout << network_logl << "\n";
    return network_logl;
}

double computeLoglikelihoodImproved(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    const Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool reuse_old_displayed_trees = reuseOldDisplayedTreesCheck(ann_network, incremental);
    if (reuse_old_displayed_trees) {
        if (ann_network.cached_logl_valid) {
            return ann_network.cached_logl;
        }
        //std::cout << "reuse displayed trees\n";
        for (size_t p = 0; p < fake_treeinfo.partition_count; ++p) { // This is in here due to reticulation prob optimization
            std::vector<DisplayedTreeData>& displayed_root_trees = ann_network.pernode_displayed_tree_data[network.root->clv_index].displayed_trees;
            size_t n_trees = ann_network.pernode_displayed_tree_data[network.root->clv_index].num_active_displayed_trees;
            for (size_t t = 0; t < n_trees; ++t) {
                assert(displayed_root_trees[t].treeLoglData.tree_logl_valid == true);
                displayed_root_trees[t].treeLoglData.tree_logprob = computeReticulationConfigLogProb(displayed_root_trees[t].treeLoglData.reticulationChoices, ann_network.reticulation_probs);
                displayed_root_trees[t].treeLoglData.tree_logprob_valid = true;
            }
        }
    } else {
        fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
        setup_pmatrices(ann_network, incremental, update_pmatrices);
        processPartitionsImproved(ann_network, incremental);
    }
    return evaluateTrees(ann_network, ann_network.network.root);
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
        TreeInfo *displayedTreeinfo = createRaxmlTreeinfo(displayed_tree, ann_network.instance);
        for (size_t p = 0; p < num_partitions; ++p) {
            displayedTreeinfo->model(p, partition_models[p]);
        }

        double tree_logl = displayedTreeinfo->loglh(0);
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

std::vector<DisplayedTreeData> extractOldTrees(AnnotatedNetwork& ann_network, Node* virtual_root) {
    std::vector<DisplayedTreeData> oldTrees;
    NodeDisplayedTreeData& nodeTrees = ann_network.pernode_displayed_tree_data[virtual_root->clv_index];
    for (size_t i = 0; i < nodeTrees.num_active_displayed_trees; ++i) {
        DisplayedTreeData& tree = nodeTrees.displayed_trees[i];
        oldTrees.emplace_back(tree);
    }
    return oldTrees;
}

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    //just for debug
    //incremental = 0;
    //update_pmatrices = 1;
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
    double bic_score = bic(ann_network, logl);
    if (bic_score == std::numeric_limits<double>::infinity()) {
        std::cout << "logl: " << logl << "\n";
        std::cout << "bic: " << bic_score << "\n";
        throw std::runtime_error("Invalid BIC score");
    }
    return bic_score;
}

}
