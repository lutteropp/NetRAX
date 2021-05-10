#include "ImprovedLoglikelihood.hpp"
#include "mpreal.h"
#include "Operation.hpp"
#include "../DebugPrintFunctions.hpp"
#include "../graph/NetworkTopology.hpp"

namespace netrax {

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

void processNodeImproved(AnnotatedNetwork& ann_network, int incremental, Node* node, std::vector<Node*>& children, const ReticulationConfigSet& extraRestrictions, bool append) {
    if (node->clv_index < ann_network.network.num_tips()) {
        return;
    }
    if (incremental && allClvsValid(ann_network.fake_treeinfo, node->clv_index)) {
        return;
    }

    if (!append) {
        ann_network.pernode_displayed_tree_data[node->clv_index].num_active_displayed_trees = 0;
    }

    if (children.size() == 0) {
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

void processPartitionsImproved(AnnotatedNetwork& ann_network, int incremental) {
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

double evaluateTreesPartition(AnnotatedNetwork& ann_network, size_t partition_idx, std::vector<TreeLoglData>& treeLoglData) {
    size_t n_trees = treeLoglData.size();

    assert(ann_network.options.likelihood_variant != LikelihoodVariant::SARAH_PSEUDO);

    if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
        mpfr::mpreal partition_lh = 0.0;
        for (size_t tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
            TreeLoglData& tree = treeLoglData[tree_idx];
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
            if (tree.tree_logprob != std::numeric_limits<double>::infinity()) {
                partition_logl = std::max(partition_logl, tree.tree_logprob + tree.tree_partition_logl[partition_idx]);
            }
        }
        ann_network.fake_treeinfo->partition_loglh[partition_idx] = partition_logl;
        return partition_logl;
    }
}

double evaluateTreesPartition(AnnotatedNetwork& ann_network, size_t partition_idx, NodeDisplayedTreeData& nodeDisplayedTreeData) {
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

    if (network_logl.toDouble() == -std::numeric_limits<double>::infinity()) {
        throw std::runtime_error("Invalid network likelihood: negative infinity \n");
    }
    ann_network.cached_logl = network_logl.toDouble();
    ann_network.cached_logl_valid = true;
    return ann_network.cached_logl;
}

double computeLoglikelihoodImproved(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    const Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool reuse_old_displayed_trees = reuseOldDisplayedTreesCheck(ann_network, incremental);
    if (reuse_old_displayed_trees) {
        if (ann_network.cached_logl_valid) {
            return ann_network.cached_logl;
        }
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
        if (update_pmatrices) {
            pllmod_treeinfo_update_prob_matrices(ann_network.fake_treeinfo, !incremental);
        }
        processPartitionsImproved(ann_network, incremental);
    }
    return evaluateTrees(ann_network, ann_network.network.root);
}

}