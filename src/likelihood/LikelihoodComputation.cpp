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

void printOperation(pll_operation_t& op) {
    std::cout << "parent_clv_index: " << op.parent_clv_index << "\n";
    std::cout << "parent_scaler_index: " << op.parent_scaler_index << "\n";
    std::cout << "child1_clv_index: " << op.child1_clv_index << "\n";
    std::cout << "child1_scaler_index: " << op.child1_scaler_index << "\n";
    std::cout << "child1_matrix_index: " << op.child1_matrix_index << "\n";
    std::cout << "child2_clv_index: " << op.child1_clv_index << "\n";
    std::cout << "child2_scaler_index: " << op.child1_scaler_index << "\n";
    std::cout << "child2_matrix_index: " << op.child1_matrix_index << "\n";
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
        ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context,
                                    treeAtRoot.treeLoglData.tree_partition_logl.data(),
                                    ann_network.fake_treeinfo->partition_count,
                                    PLLMOD_COMMON_REDUCE_SUM);
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            assert(treeAtRoot.treeLoglData.tree_partition_logl[p] != -std::numeric_limits<double>::infinity());
            assert(treeAtRoot.treeLoglData.tree_partition_logl[p] <= 0.0);
        }
    }

    treeAtRoot.treeLoglData.tree_logl_valid = true;
    treeAtRoot.treeLoglData.tree_logprob = computeReticulationConfigLogProb(treeAtRoot.treeLoglData.reticulationChoices, ann_network.reticulation_probs);
    treeAtRoot.treeLoglData.tree_logprob_valid = true;
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

void add_tree_single(AnnotatedNetwork& ann_network, size_t clv_index, pll_operation_t& op, DisplayedTreeData& childTree, const ReticulationConfigSet& reticulationChoices) {
    add_displayed_tree(ann_network, clv_index);
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[clv_index];
    size_t fake_clv_index = ann_network.network.nodes.size();
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
    tree.treeLoglData.reticulationChoices = reticulationChoices;
}

void add_tree_both(AnnotatedNetwork& ann_network, size_t clv_index, pll_operation_t& op, DisplayedTreeData& leftTree, DisplayedTreeData& rightTree, const ReticulationConfigSet& reticulationChoices) {
    add_displayed_tree(ann_network, clv_index);
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[clv_index];
    size_t fake_clv_index = ann_network.network.nodes.size();
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
        double* right_clv = rightTree.clv_vector[partition_idx];
        unsigned int* right_scaler = rightTree.scale_buffer[partition_idx];
        assert(leftTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], left_clv));
        assert(rightTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], right_clv));
        pll_update_partials_single(partition, &op, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
        assert(!single_clv_is_all_zeros(ann_network.partition_clv_ranges[partition_idx], parent_clv));
    }
    tree.treeLoglData.reticulationChoices = reticulationChoices;
}

unsigned int processNodeImprovedSingleChild(AnnotatedNetwork& ann_network, Node* node, Node* child, const ReticulationConfigSet& extraRestrictions) {
    assert(node);
    assert(child);
    pll_operation_t op = buildOperation(ann_network.network, node, child, nullptr, ann_network.network.nodes.size(), ann_network.network.edges.size());
    ReticulationConfigSet restrictionsSet = getRestrictionsToTakeNeighbor(ann_network, node, child);
    if (!extraRestrictions.configs.empty()) {
        restrictionsSet = combineReticulationChoices(restrictionsSet, extraRestrictions);
    }

    NodeDisplayedTreeData& displayed_trees_child = ann_network.pernode_displayed_tree_data[child->clv_index];
    for (size_t i = 0; i < displayed_trees_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& childTree = displayed_trees_child.displayed_trees[i];
        if (reticulationConfigsCompatible(childTree.treeLoglData.reticulationChoices, restrictionsSet)) {
            ReticulationConfigSet reticulationChoices = combineReticulationChoices(childTree.treeLoglData.reticulationChoices, restrictionsSet);
            add_tree_single(ann_network, node->clv_index, op, childTree, reticulationChoices);
        }
    }
    return displayed_trees_child.num_active_displayed_trees;
}

ReticulationConfigSet getTreeConfig(AnnotatedNetwork& ann_network, size_t tree_idx) {
    std::vector<ReticulationState> reticulationChoicesVector(ann_network.options.max_reticulations);
    ReticulationConfigSet reticulationChoices(ann_network.options.max_reticulations);
    reticulationChoices.configs.emplace_back(reticulationChoicesVector);
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        if (tree_idx & (1 << i)) {
            reticulationChoices.configs[0][i] = ReticulationState::TAKE_SECOND_PARENT;
        } else {
            reticulationChoices.configs[0][i] = ReticulationState::TAKE_FIRST_PARENT;
        }
    }
    return reticulationChoices;
}

ReticulationConfigSet deadNodeSettings(AnnotatedNetwork& ann_network, const NodeDisplayedTreeData& displayed_trees, Node* parent, Node* child) {
    // Return all configurations in which the node which the displayed trees belong to would have no displayed tree, and thus be a dead node
    ReticulationConfigSet res(ann_network.options.max_reticulations);

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
        ReticulationConfigSet reticulationChoices = getTreeConfig(ann_network, tree_idx);
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
                ReticulationConfigSet reticulationChoices = combineReticulationChoices(leftTree.treeLoglData.reticulationChoices, rightTree.treeLoglData.reticulationChoices);
                reticulationChoices = combineReticulationChoices(reticulationChoices, restrictionsBothSet);
                add_tree_both(ann_network, node->clv_index, op_both, leftTree, rightTree, reticulationChoices);
                num_trees_added++;
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
            add_tree_single(ann_network, node->clv_index, op_left_only, leftTree, leftOnlyConfigs);
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
            add_tree_single(ann_network, node->clv_index, op_right_only, rightTree, rightOnlyConfigs);
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

void processNodeImproved(AnnotatedNetwork& ann_network, int incremental, Node* node, std::vector<Node*>& children, const ReticulationConfigSet& extraRestrictions, bool append) {
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
            //std::cout << "thread " << ParallelContext::local_proc_id() << " tree " << tree_idx << " partition " << partition_idx << " logl: " << tree.tree_partition_logl[partition_idx] << "\n";
            assert(tree.tree_logl_valid);
            assert(tree.tree_logprob_valid);
            assert(tree.tree_partition_logl[partition_idx] != 0.0);
            assert(tree.tree_partition_logl[partition_idx] != -std::numeric_limits<double>::infinity());
            assert(tree.tree_partition_logl[partition_idx] < 0.0);
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
            //std::cout << "thread " << ParallelContext::local_proc_id() << " tree " << tree_idx << " partition " << partition_idx << " logl: " << tree.tree_partition_logl[partition_idx] << "\n";
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

void printDisplayedTreesChoices(AnnotatedNetwork& ann_network, size_t partition_idx, Node* virtualRoot) {
    NodeDisplayedTreeData& nodeData = ann_network.pernode_displayed_tree_data[virtualRoot->clv_index];
    for (size_t i = 0; i < nodeData.num_active_displayed_trees; ++i) {
        printReticulationChoices(nodeData.displayed_trees[i].treeLoglData.reticulationChoices);
    }
}

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
        if (update_pmatrices) {
            pllmod_treeinfo_update_prob_matrices(ann_network.fake_treeinfo, !incremental);
        }
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
    if (ann_network.options.likelihood_variant == LikelihoodVariant::SARAH_PSEUDO) {
        return computePseudoLoglikelihood(ann_network, incremental, update_pmatrices);
    } else {
        return computeLoglikelihoodImproved(ann_network, incremental, update_pmatrices);
    }
    //return computeLoglikelihoodNaiveUtree(ann_network, incremental, update_pmatrices);
}

}
