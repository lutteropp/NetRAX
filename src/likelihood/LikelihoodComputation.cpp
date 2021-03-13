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
#include "mpreal.h"

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
        if (reticulationConfigsCompatible(reticulationChoices, data.displayed_trees[i].reticulationChoices)) {
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
    // all these reticulation choices led to the same tree, thus it is safe to simply use the first one for detecting which nodes to skip...
    for (size_t i = 0; i < reticulationChoices.configs[0].size(); ++i) { // apply the reticulation choices
        setReticulationState(ann_network.network, i, reticulationChoices.configs[0][i]);
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
    treeAtRoot.tree_logprob = computeReticulationConfigLogProb(treeAtRoot.reticulationChoices, ann_network.reticulation_probs);
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

unsigned int processNodeImprovedSingleChild(AnnotatedNetwork& ann_network, unsigned int partition_idx, ClvRangeInfo &clvInfo, ScaleBufferRangeInfo &scaleBufferInfo, Node* node, Node* child, const ReticulationConfigSet& extraRestrictions) {
    assert(node);
    assert(child);

    unsigned int num_trees_added = 0;
    pll_operation_t op = buildOperation(ann_network.network, node, child, nullptr, ann_network.network.nodes.size(), ann_network.network.edges.size());
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[partition_idx][node->clv_index];
    NodeDisplayedTreeData& displayed_trees_child = ann_network.pernode_displayed_tree_data[partition_idx][child->clv_index];
    size_t fake_clv_index = ann_network.network.nodes.size();
    pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];

    ReticulationConfigSet restrictionsSet = getRestrictionsToTakeNeighbor(ann_network, node, child);

    if (!extraRestrictions.configs.empty()) {
        restrictionsSet = combineReticulationChoices(restrictionsSet, extraRestrictions);
    }

    for (size_t i = 0; i < displayed_trees_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& childTree = displayed_trees_child.displayed_trees[i];
        if (!reticulationConfigsCompatible(childTree.reticulationChoices, restrictionsSet)) {
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
        tree.reticulationChoices = combineReticulationChoices(childTree.reticulationChoices, restrictionsSet);

        if (node == ann_network.network.root) { // if we are at the root node, we also need to compute loglikelihood
            computeDisplayedTreeLoglikelihood(ann_network, partition_idx, tree, node);
        } /*else { // this is just for debug
            computeDisplayedTreeLoglikelihood(ann_network, partition_idx, tree, node);
        }*/
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
            if (reticulationConfigsCompatible(reticulationChoices, displayed_trees.displayed_trees[i].reticulationChoices)) {
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

unsigned int processNodeImprovedTwoChildren(AnnotatedNetwork& ann_network, unsigned int partition_idx, ClvRangeInfo &clvInfo, ScaleBufferRangeInfo &scaleBufferInfo, Node* node, Node* left_child, Node* right_child, const ReticulationConfigSet& extraRestrictions) {
    unsigned int num_trees_added = 0;
    size_t fake_clv_index = ann_network.network.nodes.size();
    
    pll_operation_t op_both = buildOperation(ann_network.network, node, left_child, right_child, ann_network.network.nodes.size(), ann_network.network.edges.size());
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[partition_idx][node->clv_index];
    NodeDisplayedTreeData& displayed_trees_left_child = ann_network.pernode_displayed_tree_data[partition_idx][left_child->clv_index];
    NodeDisplayedTreeData& displayed_trees_right_child = ann_network.pernode_displayed_tree_data[partition_idx][right_child->clv_index];
    pll_partition_t* partition = ann_network.fake_treeinfo->partitions[partition_idx];

    ReticulationConfigSet restrictionsBothSet = getRestrictionsToTakeNeighbor(ann_network, node, left_child);
    restrictionsBothSet = combineReticulationChoices(restrictionsBothSet, getRestrictionsToTakeNeighbor(ann_network, node, right_child));

    if (!extraRestrictions.configs.empty()) {
        restrictionsBothSet = combineReticulationChoices(restrictionsBothSet, extraRestrictions);
    }

    // left child and right child are not always about different reticulations... It can be that one reticulation affects both children.
    // It can even happen that there is a displayed tree for one child, that has no matching displaying tree on the other side (in terms of chosen reticulations). In this case, we have a dead node situation...
    
    // combine both children - here, both children are active
    for (size_t i = 0; i < displayed_trees_left_child.num_active_displayed_trees; ++i) {
        DisplayedTreeData& leftTree = displayed_trees_left_child.displayed_trees[i];
        if (!reticulationConfigsCompatible(leftTree.reticulationChoices, restrictionsBothSet)) {
            continue;
        }
        for (size_t j = 0; j < displayed_trees_right_child.num_active_displayed_trees; ++j) {
            DisplayedTreeData& rightTree = displayed_trees_right_child.displayed_trees[j];
            if (!reticulationConfigsCompatible(rightTree.reticulationChoices, restrictionsBothSet)) {
                continue;
            }
            if (reticulationConfigsCompatible(leftTree.reticulationChoices, rightTree.reticulationChoices)) {
                displayed_trees.add_displayed_tree(clvInfo, scaleBufferInfo, ann_network.options.max_reticulations);
                num_trees_added++;
                DisplayedTreeData& newDisplayedTree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];

                double* parent_clv = newDisplayedTree.clv_vector;
                unsigned int* parent_scaler = newDisplayedTree.scale_buffer;
                double* left_clv = leftTree.clv_vector;
                unsigned int* left_scaler = leftTree.scale_buffer;
                double* right_clv = rightTree.clv_vector;
                unsigned int* right_scaler = rightTree.scale_buffer;
                pll_update_partials_single(partition, &op_both, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
                newDisplayedTree.reticulationChoices = combineReticulationChoices(leftTree.reticulationChoices, rightTree.reticulationChoices);
                newDisplayedTree.reticulationChoices = combineReticulationChoices(newDisplayedTree.reticulationChoices, restrictionsBothSet);
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
        ReticulationConfigSet leftOnlyConfigs = getReticulationChoicesThisOnly(ann_network, leftTree.reticulationChoices, right_child_dead_settings, node, left_child, right_child);
        if (!extraRestrictions.configs.empty()) {
            leftOnlyConfigs = combineReticulationChoices(leftOnlyConfigs, extraRestrictions);
        }

        if (!leftOnlyConfigs.configs.empty()) {
            displayed_trees.add_displayed_tree(clvInfo, scaleBufferInfo, ann_network.options.max_reticulations);
            DisplayedTreeData& tree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];
            double* parent_clv = tree.clv_vector;
            unsigned int* parent_scaler = tree.scale_buffer;
            double* left_clv = leftTree.clv_vector;
            unsigned int* left_scaler = leftTree.scale_buffer;
            double* right_clv = partition->clv[fake_clv_index];
            unsigned int* right_scaler = nullptr;
            pll_update_partials_single(partition, &op_left_only, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
            tree.reticulationChoices = leftOnlyConfigs;
            if (node == ann_network.network.root) { // if we are at the root node, we also need to compute loglikelihood
                computeDisplayedTreeLoglikelihood(ann_network, partition_idx, tree, node);
            }
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
        ReticulationConfigSet rightOnlyConfigs = getReticulationChoicesThisOnly(ann_network, rightTree.reticulationChoices, left_child_dead_settings, node, right_child, left_child);
        if (!extraRestrictions.configs.empty()) {
            rightOnlyConfigs = combineReticulationChoices(rightOnlyConfigs, extraRestrictions);
        }
        if (!rightOnlyConfigs.configs.empty()) {
            displayed_trees.add_displayed_tree(clvInfo, scaleBufferInfo, ann_network.options.max_reticulations);
            DisplayedTreeData& tree = displayed_trees.displayed_trees[displayed_trees.num_active_displayed_trees-1];
            double* parent_clv = tree.clv_vector;
            unsigned int* parent_scaler = tree.scale_buffer;
            double* left_clv = rightTree.clv_vector;
            unsigned int* left_scaler = rightTree.scale_buffer;
            double* right_clv = partition->clv[fake_clv_index];
            unsigned int* right_scaler = nullptr;
            pll_update_partials_single(partition, &op_right_only, 1, parent_clv, left_clv, right_clv, parent_scaler, left_scaler, right_scaler);
            tree.reticulationChoices = rightOnlyConfigs;
            num_trees_added++;
        }
    }

    return num_trees_added;
}

void processNodeImproved(AnnotatedNetwork& ann_network, unsigned int partition_idx, int incremental, ClvRangeInfo &clvInfo, ScaleBufferRangeInfo &scaleBufferInfo, Node* node, std::vector<Node*>& children, const ReticulationConfigSet& extraRestrictions, bool append = false) {
    if (node->clv_index < ann_network.network.num_tips()) {
        //assert(ann_network.fake_treeinfo->clv_valid[partition_idx][node->clv_index]);
        return;
    }
    NodeDisplayedTreeData& displayed_trees = ann_network.pernode_displayed_tree_data[partition_idx][node->clv_index];
    if (incremental && ann_network.fake_treeinfo->clv_valid[partition_idx][node->clv_index]) {
        return;
    }

    if (!append) {
        displayed_trees.num_active_displayed_trees = 0;
    }

    if (children.size() == 1) {
        processNodeImprovedSingleChild(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, children[0], extraRestrictions);
    } else {
        assert(children.size() == 2);
        Node* left_child = children[0];
        Node* right_child = children[1];
        processNodeImprovedTwoChildren(ann_network, partition_idx, clvInfo, scaleBufferInfo, node, left_child, right_child, extraRestrictions);
    }

    ann_network.fake_treeinfo->clv_valid[partition_idx][node->clv_index] = 1;
    /*std::cout << "Node " << node->clv_index << " has been processed, displayed trees at the node are:\n";
    for (size_t i = 0; i < displayed_trees.num_active_displayed_trees; ++i) {
        printReticulationChoices(displayed_trees.displayed_trees[i].reticulationChoices);
    }*/
}

void processPartitionImproved(AnnotatedNetwork& ann_network, unsigned int partition_idx, int incremental) {
    //std::cout << "\nNEW PROCESS PARTITION_IMPROVED!!!\n";
    std::vector<bool> seen(ann_network.network.num_nodes(), false);
    ClvRangeInfo clvInfo = get_clv_range(ann_network.fake_treeinfo->partitions[partition_idx]);
    ScaleBufferRangeInfo scaleBufferInfo = get_scale_buffer_range(ann_network.fake_treeinfo->partitions[partition_idx]);
    
    for (size_t i = 0; i < ann_network.travbuffer.size(); ++i) {
        Node* actNode = ann_network.travbuffer[i];
        std::vector<Node*> children = getChildren(ann_network.network, actNode);
        processNodeImproved(ann_network, partition_idx, incremental, clvInfo, scaleBufferInfo, actNode, children, {});
    }

    for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[partition_idx][ann_network.network.root->clv_index].num_active_displayed_trees; ++i) {
        DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[partition_idx][ann_network.network.root->clv_index].displayed_trees[i];
        computeDisplayedTreeLoglikelihood(ann_network, partition_idx, tree, ann_network.network.root);
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
        all_clvs_valid &= fake_treeinfo.clv_valid[i][ann_network.network.root->clv_index];
    }
    return all_clvs_valid;
}

DisplayedTreeData& getMatchingDisplayedTreeAtNode(AnnotatedNetwork& ann_network, unsigned int partition_idx, unsigned int node_clv_index, const ReticulationConfigSet& queryChoices) {
    for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[partition_idx][node_clv_index].num_active_displayed_trees; ++i) {
        if (reticulationConfigsCompatible(queryChoices, ann_network.pernode_displayed_tree_data[partition_idx][node_clv_index].displayed_trees[i].reticulationChoices)) {
            return ann_network.pernode_displayed_tree_data[partition_idx][node_clv_index].displayed_trees[i];
        }
    }
    throw std::runtime_error("No compatible displayed tree data found");
}

const OldTreeLoglData& getMatchingOldTree(AnnotatedNetwork& ann_network, const std::vector<OldTreeLoglData>& oldTrees, const ReticulationConfigSet& queryChoices) {
    for (size_t i = 0; i < oldTrees.size(); ++i) {
        if (reticulationConfigsCompatible(queryChoices, oldTrees[i].reticulationChoices)) {
            return oldTrees[i];
        }
    }
    throw std::runtime_error("No compatible old tree data found");
}

double evaluateTrees(AnnotatedNetwork &ann_network, Node* virtual_root) {
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    mpfr::mpreal network_logl = 0.0;

    for (size_t partition_idx = 0; partition_idx < fake_treeinfo.partition_count; ++partition_idx) {
        fake_treeinfo.active_partition = partition_idx;
        std::vector<DisplayedTreeData>& displayed_root_trees = ann_network.pernode_displayed_tree_data[partition_idx][virtual_root->clv_index].displayed_trees;
        size_t n_trees = ann_network.pernode_displayed_tree_data[partition_idx][virtual_root->clv_index].num_active_displayed_trees;

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
                    std::cout << "displayed trees stored at node " << virtual_root->clv_index << ":\n";
                    for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[0][virtual_root->clv_index].num_active_displayed_trees; ++j) {
                        printReticulationChoices(ann_network.pernode_displayed_tree_data[0][virtual_root->clv_index].displayed_trees[j].reticulationChoices);
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
    if (network_logl.toDouble() == -std::numeric_limits<double>::infinity()) {
        std::cout << exportDebugInfo(ann_network) << "\n";
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            std::cout << "reticulation node " << ann_network.network.reticulation_nodes[i]->clv_index << " has first parent " << getReticulationFirstParent(ann_network.network, ann_network.network.reticulation_nodes[i])->clv_index << "\n";
        }
        for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
            std::cout << "displayed trees stored at node " << i << ":\n";
            for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[0][i].num_active_displayed_trees; ++j) {
                printReticulationChoices(ann_network.pernode_displayed_tree_data[0][i].displayed_trees[j].reticulationChoices);
            }
        }
        throw std::runtime_error("Invalid network likelihood: negative infinity \n");
    }
    ann_network.cached_logl = network_logl.toDouble();
    ann_network.cached_logl_valid = true;
    return ann_network.cached_logl;
}

bool isActiveBranch(AnnotatedNetwork& ann_network, const DisplayedTreeData& displayedTree, unsigned int pmatrix_index) {
    Node* edge_source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* edge_target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);

    ReticulationConfigSet restrictions = getRestrictionsToTakeNeighbor(ann_network, edge_source, edge_target);
    return reticulationConfigsCompatible(restrictions, displayedTree.reticulationChoices);
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
    return res;
}

std::vector<PathToVirtualRoot> getPathsToVirtualRoot(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back) {
    std::vector<PathToVirtualRoot> res;
    // naive version here: Go over all displayed trees, compute pathToVirtualRoot for each of them, and then later on kick out duplicate paths...

    NodeDisplayedTreeData& oldDisplayedTrees = ann_network.pernode_displayed_tree_data[0][old_virtual_root->clv_index];
    for (size_t i = 0; i < oldDisplayedTrees.num_active_displayed_trees; ++i) {
        PathToVirtualRoot ptvr(ann_network.options.max_reticulations);
        setReticulationParents(ann_network.network, oldDisplayedTrees.displayed_trees[i].reticulationChoices.configs[0]);
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

    std::cout << "paths from old virtual root " << old_virtual_root->clv_index << " to virtual root " << new_virtual_root->clv_index << ":\n";
    for (size_t i = 0; i < res.size(); ++i) {
        printPathToVirtualRoot(res[i]);
    }

    return res;
}

void updateCLVsVirtualRerootTrees(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back) {
    assert(old_virtual_root);
    assert(new_virtual_root);

    // 1.) for all paths from retNode to new_virtual_root:
    //     1.1) Collect the reticulation nodes encountered on the path, build exta restrictions storing the reticulation configurations used
    //     1.2) update CLVs on that path, using extra restrictions and append mode
    std::vector<PathToVirtualRoot> paths = getPathsToVirtualRoot(ann_network, old_virtual_root, new_virtual_root, new_virtual_root_back);

    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        ClvRangeInfo clvInfo = get_clv_range(ann_network.fake_treeinfo->partitions[partition_idx]);
        ScaleBufferRangeInfo scaleBufferInfo = get_scale_buffer_range(ann_network.fake_treeinfo->partitions[partition_idx]);
        for (size_t p = 0; p < paths.size(); ++p) {
            for (size_t i = 0; i < paths[p].path.size(); ++i) {
                bool appendMode = (p > 0);
                processNodeImproved(ann_network, partition_idx, 0, clvInfo, scaleBufferInfo, paths[p].path[i], paths[p].children[i], paths[p].reticulationChoices, appendMode);
            }
        }
    }    
}

void printDisplayedTreesChoices(AnnotatedNetwork& ann_network, Node* virtualRoot) {
    NodeDisplayedTreeData& nodeData = ann_network.pernode_displayed_tree_data[0][virtualRoot->clv_index];
    for (size_t i = 0; i < nodeData.num_active_displayed_trees; ++i) {
        printReticulationChoices(nodeData.displayed_trees[i].reticulationChoices);
    }
}

double computeLoglikelihoodBrlenOpt(AnnotatedNetwork &ann_network, const std::vector<std::vector<OldTreeLoglData> >& oldTrees, unsigned int pmatrix_index, int incremental, int update_pmatrices) {
    Node* source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    assert(reuseOldDisplayedTreesCheck(ann_network, incremental)); // TODO: Doesn't this need the virtual_root pointer, too?
    setup_pmatrices(ann_network, incremental, update_pmatrices);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        pll_partition_t* partition = ann_network.fake_treeinfo->partitions[p];
        size_t n_trees = ann_network.pernode_displayed_tree_data[p][source->clv_index].num_active_displayed_trees;
        for (size_t i = 0; i < n_trees; ++i) {
            DisplayedTreeData& sourceTree = ann_network.pernode_displayed_tree_data[p][source->clv_index].displayed_trees[i];
            if (isActiveBranch(ann_network, sourceTree, pmatrix_index)) {
                std::cout << "Branch " << pmatrix_index << " is active in this displayed tree:\n";
                printReticulationChoices(sourceTree.reticulationChoices);

                DisplayedTreeData& targetTree = getMatchingDisplayedTreeAtNode(ann_network, p, target->clv_index, sourceTree.reticulationChoices);
                sourceTree.tree_logl = pll_compute_edge_loglikelihood(partition, source->clv_index, sourceTree.clv_vector, sourceTree.scale_buffer, 
                                                                target->clv_index, targetTree.clv_vector, targetTree.scale_buffer, 
                                                                pmatrix_index, ann_network.fake_treeinfo->param_indices[p], nullptr);
                sourceTree.tree_logprob = computeReticulationConfigLogProb(sourceTree.reticulationChoices, ann_network.reticulation_probs);
            } else { // for inactive branches (in dead area), we have a dead node situation. However, we don't need to recompute tree loglh here as it stays the same as it was for the old virtual root
                std::cout << "USING OLD DISPLAYED TREE\n";
                const OldTreeLoglData& oldTree = getMatchingOldTree(ann_network, oldTrees[p], sourceTree.reticulationChoices);
                assert(oldTree.tree_logl_valid);
                sourceTree.tree_logl = oldTree.tree_logl;
                assert(oldTree.tree_logprob_valid);
                sourceTree.tree_logprob = oldTree.tree_logprob;
            }
            sourceTree.tree_logl_valid = true;
            sourceTree.tree_logprob_valid = true;
        }
    }

    double network_logl = evaluateTrees(ann_network, source);

    // TODO: Remove me again, this is just for debug
    std::cout << "Displayed trees to evaluate:\n";
    printDisplayedTreesChoices(ann_network, source);
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
        for (size_t p = 0; p < fake_treeinfo.partition_count; ++p) { // TODO: Why is this needed here?
            std::vector<DisplayedTreeData>& displayed_root_trees = ann_network.pernode_displayed_tree_data[p][network.root->clv_index].displayed_trees;
            size_t n_trees = ann_network.pernode_displayed_tree_data[p][network.root->clv_index].num_active_displayed_trees;
            for (size_t t = 0; t < n_trees; ++t) {
                assert(displayed_root_trees[t].tree_logl_valid == true);
                displayed_root_trees[t].tree_logprob = computeReticulationConfigLogProb(displayed_root_trees[t].reticulationChoices, ann_network.reticulation_probs);
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
