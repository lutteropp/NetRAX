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

#include <cassert>
#include <cmath>

namespace netrax {

// TODO: Use a dirty flag to only update CLVs that are needed... actually, we might already have it. It's called clv_valid and is in the treeinfo...

void printClv(const pllmod_treeinfo_t &treeinfo, size_t clv_index, size_t partition_index) {
    size_t sites = treeinfo.partitions[partition_index]->sites;
    size_t rate_cats = treeinfo.partitions[partition_index]->rate_cats;
    size_t states = treeinfo.partitions[partition_index]->states;
    size_t states_padded = treeinfo.partitions[partition_index]->states_padded;
    std::cout << "Clv for clv_index " << clv_index << ": \n";
    for (unsigned int n = 0; n < sites; ++n) {
        for (unsigned int i = 0; i < rate_cats; ++i) {
            for (unsigned int j = 0; j < states; ++j) {
                std::cout << treeinfo.partitions[partition_index]->clv[clv_index][j + i * states_padded] << "\n";
            }
        }
    }
}

pll_operation_t buildOperationInternal(Network &network, Node *parent, Node *child1, Node *child2,
        size_t fake_clv_index, size_t fake_pmatrix_index) {
    pll_operation_t operation;
    assert(parent);
    assert(child1 || child2);
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

pll_operation_t buildOperation(Network &network, Node *actNode, Node* actParent, const std::vector<bool> &dead_nodes,
        size_t fake_clv_index, size_t fake_pmatrix_index) {
    std::vector<Node*> activeChildren = getActiveChildrenIgnoreDirections(network, actNode, actParent);
    assert(activeChildren.size() > 0);
    Node *child1 = nullptr;
    if (!dead_nodes[activeChildren[0]->clv_index]) {
        child1 = activeChildren[0];
    }
    Node *child2 = nullptr;
    if (activeChildren.size() == 2 && !dead_nodes[activeChildren[1]->clv_index]) {
        child2 = activeChildren[1];
    }
    return buildOperationInternal(network, actNode, child1, child2, fake_clv_index, fake_pmatrix_index);
}

void createOperationsPostorder(AnnotatedNetwork &ann_network, bool incremental, size_t partition_idx, Node *actNode,
        Node *parent, std::vector<pll_operation_t> &ops, size_t fake_clv_index, size_t fake_pmatrix_index,
        const std::vector<bool> &dead_nodes, const std::vector<unsigned int> *stop_indices = nullptr) {
    if (stop_indices
            && std::find(stop_indices->begin(), stop_indices->end(), actNode->clv_index) != stop_indices->end()) {
        return;
    }
    Network &network = ann_network.network;
    if (incremental && ann_network.network.num_reticulations() == 0
            && ann_network.fake_treeinfo->clv_valid[partition_idx][actNode->clv_index]) {
        return;
    }

    std::vector<Node*> activeChildren = getActiveChildrenIgnoreDirections(network, actNode, parent);
    if (activeChildren.empty()) { // nothing to do if we are at a leaf node
        return;
    }
    assert(activeChildren.size() <= 2);
    for (size_t i = 0; i < activeChildren.size(); ++i) {
        if (!dead_nodes[activeChildren[i]->clv_index]) {
            createOperationsPostorder(ann_network, incremental, partition_idx, activeChildren[i], actNode, ops,
                    fake_clv_index, fake_pmatrix_index, dead_nodes, stop_indices);
        }
    }

    pll_operation_t operation = buildOperation(network, actNode, parent, dead_nodes, fake_clv_index,
            fake_pmatrix_index);
    ops.push_back(operation);
}

void printOperationArray(const std::vector<pll_operation_t> &ops) {
    for (size_t i = 0; i < ops.size(); ++i) {
        size_t c1 = ops[i].child1_clv_index;
        size_t c2 = ops[i].child2_clv_index;
        size_t p = ops[i].parent_clv_index;

        std::cout << p << " -> " << c1 << ", " << c2 << "\n";
    }
}

std::vector<pll_operation_t> createOperations(AnnotatedNetwork &ann_network, size_t partition_idx,
        const std::vector<Node*> &parent, BlobInformation &blobInfo, unsigned int megablobIdx,
        const std::vector<bool> &dead_nodes, bool incremental) {
    Network &network = ann_network.network;
    std::vector<pll_operation_t> ops;
    size_t fake_clv_index = network.nodes.size();
    size_t fake_pmatrix_index = network.edges.size();

    // fill forbidden clv indices
    std::vector<unsigned int> stopIndices;
    for (size_t i = 0; i < blobInfo.megablob_roots.size(); ++i) {
        if (i != megablobIdx) {
            stopIndices.emplace_back(blobInfo.megablob_roots[i]->clv_index);
        }
    }

    if (blobInfo.megablob_roots[megablobIdx] == network.root) {
        // How to do the operations at the top-level root trifurcation?
        // First with root->back, then with root...
        Node *rootBack = getTargetNode(network, network.root->getLink());
        createOperationsPostorder(ann_network, incremental, partition_idx, rootBack, network.root, ops, fake_clv_index,
                fake_pmatrix_index, dead_nodes, &stopIndices);

        if (!getActiveChildrenIgnoreDirections(network, network.root, rootBack).empty()) {
            createOperationsPostorder(ann_network, incremental, partition_idx, network.root, rootBack, ops,
                    fake_clv_index, fake_pmatrix_index, dead_nodes, &stopIndices);
        } else {
            // special case: the root has a single child.
            ops.push_back(
                    buildOperationInternal(network, network.root, rootBack, nullptr, fake_clv_index,
                            fake_pmatrix_index));
            // ignore the branch length from the root to its single active child/ treat it as if it had zero branch length
            ops[ops.size() - 1].child1_matrix_index = fake_pmatrix_index;
        }
        if (ops.size() > 0) {
            assert(ops[ops.size() - 1].parent_clv_index == network.root->clv_index);
        }
    } else {
        Node *megablobRoot = blobInfo.megablob_roots[megablobIdx];
        createOperationsPostorder(ann_network, incremental, partition_idx, megablobRoot,
                parent[megablobRoot->clv_index], ops, fake_clv_index, fake_pmatrix_index, dead_nodes, &stopIndices);
    }
    //printOperationArray(ops);
    return ops;
}

Node* getActiveParent(Network &network, Node *actNode, const std::vector<Node*> &parent) {
    Node *actParent;
    if (actNode->type == NodeType::RETICULATION_NODE) {
        actParent = getReticulationActiveParent(network, actNode);
    } else {
        actParent = parent[actNode->clv_index];
    }
    return actParent;
}

std::vector<pll_operation_t> createOperationsTowardsRoot(AnnotatedNetwork &ann_network, size_t partition_idx,
        const std::vector<Node*> &parent, Node *actParent, const std::vector<bool> &dead_nodes, bool incremental) {
    Network &network = ann_network.network;

    std::vector<pll_operation_t> ops;
    if (actParent == nullptr || actParent == network.root) {
        return ops;
    }
    if (dead_nodes[actParent->clv_index]) {
        return createOperationsTowardsRoot(ann_network, partition_idx, parent, parent[actParent->clv_index], dead_nodes,
                incremental);
    }

    size_t fake_clv_index = network.nodes.size();
    size_t fake_pmatrix_index = network.edges.size();
    Node *rootBack = getTargetNode(network, network.root->getLink());

    while (actParent != network.root && actParent != rootBack) {
        if (!incremental || ann_network.network.num_reticulations() != 0
                || !ann_network.fake_treeinfo->clv_valid[partition_idx][actParent->clv_index]) {
            ops.emplace_back(
                    buildOperation(network, actParent, parent[actParent->clv_index], dead_nodes, fake_clv_index,
                            fake_pmatrix_index));
        }
        actParent = getActiveParent(network, actParent, parent);
    }

    // now, add the two operations for the root node in reverse order.
    if (!rootBack->isTip() && !getActiveChildrenIgnoreDirections(network, rootBack, network.root).empty()) {
        ops.push_back(buildOperation(network, rootBack, network.root, dead_nodes, fake_clv_index, fake_pmatrix_index));
    }

    if (!getActiveChildrenIgnoreDirections(network, network.root, rootBack).empty()) {
        ops.push_back(buildOperation(network, network.root, rootBack, dead_nodes, fake_clv_index, fake_pmatrix_index));
    } else {
        // special case: the root has a single child.
        ops.push_back(
                buildOperationInternal(network, network.root, rootBack, nullptr, fake_clv_index, fake_pmatrix_index));
        // ignore the branch length from the root to its single active child/ treat it as if it had zero branch length
        ops[ops.size() - 1].child1_matrix_index = fake_pmatrix_index;
    }

    //printOperationArray(ops);
    assert(ops[ops.size() - 1].parent_clv_index == network.root->clv_index);
    return ops;
}

std::vector<pll_operation_t> createOperationsUpdatedReticulation(AnnotatedNetwork &ann_network, size_t partition_idx,
        const std::vector<Node*> &parent, Node *actNode, const std::vector<bool> &dead_nodes, bool incremental) {
    Network &network = ann_network.network;
    std::vector<pll_operation_t> ops;

    Node *firstParent = getReticulationFirstParent(network, actNode);
    std::vector<pll_operation_t> opsFirst = createOperationsTowardsRoot(ann_network, partition_idx, parent, firstParent,
            dead_nodes, incremental);
    Node *secondParent = getReticulationSecondParent(network, actNode);
    std::vector<pll_operation_t> opsSecond = createOperationsTowardsRoot(ann_network, partition_idx, parent,
            secondParent, dead_nodes, incremental);

    // find the first entry in opsFirst which also occurs in opsSecond. We will only take opsFirst until this entry, excluding it.
    std::unordered_set<unsigned int> opsSecondRoots;
    for (size_t i = 0; i < opsSecond.size(); ++i) {
        opsSecondRoots.emplace(opsSecond[i].parent_clv_index);
    }
    for (size_t i = 0; i < opsFirst.size(); ++i) {
        if (opsSecondRoots.find(opsFirst[i].parent_clv_index) != opsSecondRoots.end()) {
            break;
        }
        ops.emplace_back(opsFirst[i]);
    }
    for (size_t i = 0; i < opsSecond.size(); ++i) {
        ops.emplace_back(opsSecond[i]);
    }

    assert(ops[ops.size() - 1].parent_clv_index == network.root->clv_index);
    //printOperationArray(ops);
    return ops;
}

std::vector<pll_operation_t> createOperations(AnnotatedNetwork &ann_network, size_t partition_idx, size_t treeIdx,
        const std::vector<bool> &dead_nodes, bool incremental) {
    Network &network = ann_network.network;

    std::vector<pll_operation_t> ops;
    size_t fake_clv_index = network.nodes.size();
    size_t fake_pmatrix_index = network.edges.size();

    Node *rootBack = getTargetNode(network, network.root->getLink());

    // How to do the operations at the top-level root trifurcation?
    // First with root->back, then with root...
    createOperationsPostorder(ann_network, incremental, partition_idx, rootBack, network.root, ops, fake_clv_index,
            fake_pmatrix_index, dead_nodes);

    if (!getActiveChildrenIgnoreDirections(network, network.root, rootBack).empty()) {
        createOperationsPostorder(ann_network, incremental, partition_idx, network.root, rootBack, ops, fake_clv_index,
                fake_pmatrix_index, dead_nodes);
    } else {
        // special case: the root has a single child.
        ops.push_back(
                buildOperationInternal(network, network.root, rootBack, nullptr, fake_clv_index, fake_pmatrix_index));
        // ignore the branch length from the root to its single active child/ treat it as if it had zero branch length
        ops[ops.size() - 1].child1_matrix_index = fake_pmatrix_index;
    }

    if (ops.size() > 0) {
        assert(ops[ops.size() - 1].parent_clv_index == network.root->clv_index);
    }
    //printOperationArray(ops);
    return ops;
}

double displayed_tree_prob(AnnotatedNetwork &ann_network, size_t tree_index, size_t partition_index) {
    Network &network = ann_network.network;
    setReticulationParents(network, tree_index);
    double logProb = 0;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        size_t active_pmatrix_idx = getReticulationActiveParentPmatrixIndex(network, network.reticulation_nodes[i]);
        double prob = ann_network.branch_probs[partition_index][active_pmatrix_idx];
        logProb += log(prob);
    }
    return exp(logProb);
}

double displayed_tree_blob_prob(AnnotatedNetwork &ann_network, size_t megablob_idx, size_t partition_index) {
    Network &network = ann_network.network;
    BlobInformation &blobInfo = ann_network.blobInfo;
    double logProb = 0;
    for (size_t i = 0; i < blobInfo.reticulation_nodes_per_megablob[megablob_idx].size(); ++i) {
        size_t active_pmatrix_idx = getReticulationActiveParentPmatrixIndex(network,
                blobInfo.reticulation_nodes_per_megablob[megablob_idx][i]);
        double prob = ann_network.branch_probs[partition_index][active_pmatrix_idx];
        logProb += log(prob);
    }
    return exp(logProb);
}

void print_clv_vector(pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx, size_t partition_idx, size_t clv_index) {
    pll_partition_t *partition = fake_treeinfo.partitions[partition_idx];
    unsigned int states_padded = partition->states_padded;
    unsigned int sites = partition->sites;
    unsigned int rate_cats = partition->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    double *clv = partition->clv[clv_index];
    std::cout << "clv vector for tree_idx " << tree_idx << " and clv_index " << clv_index << ":\n";
    for (size_t i = 0; i < clv_len; ++i) {
        std::cout << clv[i] << ",";
    }
    std::cout << "\n";
}

double compute_tree_logl(AnnotatedNetwork &ann_network, size_t tree_idx, size_t partition_idx,
        std::vector<double> *persite_logl, const std::vector<Node*> &parent, Node *startNode = nullptr,
        bool incremental = true) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;

    std::vector<bool> dead_nodes(network.num_nodes(), false);

    fill_dead_nodes_recursive(network, nullptr, network.root, dead_nodes);

// Create pll_operations_t array for the current displayed tree
    std::vector<pll_operation_t> ops;
    if (startNode) {
        ops = createOperationsUpdatedReticulation(ann_network, partition_idx, parent, startNode, dead_nodes,
                incremental);
    } else {
        ops = createOperations(ann_network, partition_idx, tree_idx, dead_nodes, incremental);
    }
    unsigned int ops_count = ops.size();

    Node *ops_root;
    if (ops_count > 0) {
        ops_root = network.nodes_by_index[ops[ops.size() - 1].parent_clv_index];
        assert(ops_root == network.root);
// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
        pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);
    } else {
        ops_root = network.root;
    }
    Node *rootBack = getTargetNode(network, network.root->getLink());
    double tree_partition_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx],
            ops_root->clv_index, ops_root->scaler_index, rootBack->clv_index, rootBack->scaler_index,
            ops_root->getLink()->edge_pmatrix_index, fake_treeinfo.param_indices[partition_idx],
            persite_logl->empty() ? nullptr : persite_logl->data());
    return tree_partition_logl;
}

void compute_tree_logl_blobs(AnnotatedNetwork &ann_network, bool incremental, const std::vector<Node*> &parent,
        pllmod_treeinfo_t &fake_treeinfo, size_t megablob_idx, size_t partition_idx, std::vector<double> *persite_logl,
        Node *startNode = nullptr) {
    Network &network = ann_network.network;
    BlobInformation &blobInfo = ann_network.blobInfo;

    std::vector<bool> dead_nodes(network.num_nodes(), false);
    fill_dead_nodes_recursive(network, nullptr, network.root, dead_nodes);
// Create pll_operations_t array for the current displayed tree
    std::vector<pll_operation_t> ops;
    if (startNode) {
        ops = createOperationsUpdatedReticulation(ann_network, partition_idx, parent, startNode, dead_nodes,
                incremental);
    } else {
        ops = createOperations(ann_network, partition_idx, parent, blobInfo, megablob_idx, dead_nodes, incremental);
    }
    unsigned int ops_count = ops.size();
    if (ops_count == 0) {
        return;
    }
    Node *ops_root = network.nodes_by_index[ops[ops.size() - 1].parent_clv_index];

// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
    pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);

    if (persite_logl != nullptr) {
        double tree_partition_logl;
        if (ops_root == network.root) {
            Node *rootBack = getTargetNode(network, ops_root->getLink());
            tree_partition_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx],
                    ops_root->clv_index, ops_root->scaler_index, rootBack->clv_index, rootBack->scaler_index,
                    ops_root->getLink()->edge_pmatrix_index, fake_treeinfo.param_indices[partition_idx],
                    persite_logl->empty() ? nullptr : persite_logl->data());
        } else {
            tree_partition_logl = pll_compute_root_loglikelihood(fake_treeinfo.partitions[partition_idx],
                    ops_root->clv_index, ops_root->scaler_index, fake_treeinfo.param_indices[partition_idx],
                    persite_logl->empty() ? nullptr : persite_logl->data());
        }

        //std::cout << "tree_partition_logl for megablob root clv index " << ops_root->clv_index << ", tree_idx " << tree_idx << ": " << tree_partition_logl << "\n";
    }
}

// TODO: Add bool incremental...
void setup_pmatrices(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    /* NOTE: in unlinked brlen mode, up-to-date brlens for partition p
     * have to be prefetched to treeinfo->branch_lengths[p] !!! */
    bool collect_brlen = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? false : true);
    if (collect_brlen) {
        for (size_t i = 0; i < network.edges.size() + 1; ++i) { // +1 for the the fake entry
            fake_treeinfo.branch_lengths[0][i] = 0.0;
            ann_network.branch_probs[0][i] = 1.0;
        }
        for (size_t i = 0; i < network.num_branches(); ++i) {
            fake_treeinfo.branch_lengths[0][network.edges[i].pmatrix_index] = network.edges[i].length;
            ann_network.branch_probs[0][network.edges[i].pmatrix_index] = network.edges[i].prob;
        }
        if (update_pmatrices) {
            pllmod_treeinfo_update_prob_matrices(&fake_treeinfo, !incremental);
        }
    }
}

struct BestPersiteLoglikelihoodData {
    double best_site_logl;
    // for each reticulation: in how many displayed trees has it been taken/ not taken to get to this site_logl?
    std::vector<unsigned int> first_parent_taken_for_best_cnt;

    BestPersiteLoglikelihoodData(unsigned int reticulation_count) :
            best_site_logl(-std::numeric_limits<double>::infinity()), first_parent_taken_for_best_cnt(
                    reticulation_count, 0) {
    }
};

void updateBestPersiteLoglikelihoods(unsigned int treeIdx, unsigned int num_reticulations, unsigned int numSites,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, const std::vector<double> &persite_logl) {
    for (size_t s = 0; s < numSites; ++s) {
        if (best_persite_logl_network[s].best_site_logl < persite_logl[s]) {
            std::fill(best_persite_logl_network[s].first_parent_taken_for_best_cnt.begin(),
                    best_persite_logl_network[s].first_parent_taken_for_best_cnt.end(), 0);
            best_persite_logl_network[s].best_site_logl = persite_logl[s];
        }
        if (best_persite_logl_network[s].best_site_logl == persite_logl[s]) {
            for (size_t r = 0; r < num_reticulations; ++r) {
                if (treeIdx & (1 << r)) {
                    best_persite_logl_network[s].first_parent_taken_for_best_cnt[r]++;
                }
            }
        }
    }
}

void updateBestPersiteLoglikelihoodsBlobs(Network &network, const BlobInformation &blobInfo, unsigned int megablob_idx,
        unsigned int treeIdx, unsigned int numSites,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, const std::vector<double> &persite_logl) {
    for (size_t s = 0; s < numSites; ++s) {
        if (best_persite_logl_network[s].best_site_logl < persite_logl[s]) {
            std::fill(best_persite_logl_network[s].first_parent_taken_for_best_cnt.begin(),
                    best_persite_logl_network[s].first_parent_taken_for_best_cnt.end(), 0);
            best_persite_logl_network[s].best_site_logl = persite_logl[s];
        }
        if (best_persite_logl_network[s].best_site_logl == persite_logl[s]) {
            size_t num_reticulations = blobInfo.reticulation_nodes_per_megablob[megablob_idx].size();
            for (size_t r = 0; r < num_reticulations; ++r) {
                if (treeIdx & (1 << r)) {
                    size_t retIdxInNetwork = 0;
                    unsigned int retClVIdx = blobInfo.reticulation_nodes_per_megablob[megablob_idx][r]->clv_index;
                    for (size_t i = 0; i < network.num_reticulations(); ++i) {
                        if (network.reticulation_nodes[i]->clv_index == retClVIdx) {
                            retIdxInNetwork = i;
                            break;
                        }
                    }
                    best_persite_logl_network[s].first_parent_taken_for_best_cnt[retIdxInNetwork]++;
                }
            }
        }
    }
}

void update_total_taken(std::vector<unsigned int> &totalTaken, std::vector<unsigned int> &totalNotTaken,
        bool unlinked_mode, unsigned int numSites, unsigned int num_reticulations,
        const std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network) {
    if (unlinked_mode) {
        std::fill(totalTaken.begin(), totalTaken.end(), 0);
        std::fill(totalNotTaken.begin(), totalNotTaken.end(), 0);
    }
    size_t n_trees = 1 << num_reticulations;
    for (size_t s = 0; s < numSites; ++s) {
        for (size_t r = 0; r < num_reticulations; ++r) {
            totalTaken[r] += best_persite_logl_network[s].first_parent_taken_for_best_cnt[r];
            totalNotTaken[r] += n_trees - best_persite_logl_network[s].first_parent_taken_for_best_cnt[r];
        }
    }
}

bool update_probs(AnnotatedNetwork &ann_network, unsigned int partitionIdx, const std::vector<unsigned int> &totalTaken,
        const std::vector<unsigned int> &totalNotTaken) {
    Network &network = ann_network.network;
    bool reticulationProbsHaveChanged = false;
    for (size_t r = 0; r < network.num_reticulations(); ++r) {
        double newProb = (double) totalTaken[r] / (totalTaken[r] + totalNotTaken[r]); // Percentage of sites that were maximized when taking this reticulation

        size_t first_parent_pmatrix_index = getReticulationFirstParentPmatrixIndex(network,
                network.reticulation_nodes[r]);
        double oldProb = ann_network.branch_probs[partitionIdx][first_parent_pmatrix_index];

        if (newProb != oldProb) {
            size_t second_parent_pmatrix_index = getReticulationSecondParentPmatrixIndex(network,
                    network.reticulation_nodes[r]);
            ann_network.branch_probs[partitionIdx][first_parent_pmatrix_index] = newProb;
            ann_network.branch_probs[partitionIdx][second_parent_pmatrix_index] = 1.0 - newProb;
            reticulationProbsHaveChanged = true;
        }
    }
    return reticulationProbsHaveChanged;
}

void merge_tree_clvs(const std::vector<std::pair<double, std::vector<double>>> &tree_clvs, pll_partition_t *partition,
        unsigned int rootCLVIndex) {
    unsigned int states_padded = partition->states_padded;
    unsigned int sites = partition->sites;
    unsigned int rate_cats = partition->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    double *clv = partition->clv[rootCLVIndex];

    for (unsigned int i = 0; i < clv_len; ++i) {
        clv[i] = 0;
        for (unsigned int k = 0; k < tree_clvs.size(); ++k) {
            clv[i] += tree_clvs[k].first * tree_clvs[k].second[i];
        }
    }
}

std::vector<double> compute_persite_lh_blobs(AnnotatedNetwork &ann_network, unsigned int partitionIdx,
        const std::vector<Node*> &parent, bool unlinked_mode, bool update_reticulation_probs, unsigned int numSites,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, bool incremental) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool useGrayCode = ann_network.options.use_graycode;
    BlobInformation &blobInfo = ann_network.blobInfo;

    unsigned int states_padded = fake_treeinfo.partitions[partitionIdx]->states_padded;
    unsigned int sites = fake_treeinfo.partitions[partitionIdx]->sites;
    unsigned int rate_cats = fake_treeinfo.partitions[partitionIdx]->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    std::vector<double> persite_lh_network(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
    // Iterate over all megablobs in a bottom-up manner
    for (size_t megablob_idx = 0; megablob_idx < blobInfo.megablob_roots.size(); ++megablob_idx) {
        unsigned int megablobRootClvIdx = blobInfo.megablob_roots[megablob_idx]->clv_index;
        if (incremental && megablobRootClvIdx != network.root->clv_index
                && fake_treeinfo.clv_valid[partitionIdx][megablobRootClvIdx] && !update_reticulation_probs) {
            continue;
        }

        size_t n_trees = 1 << blobInfo.reticulation_nodes_per_megablob[megablob_idx].size();
        // iterate over all displayed trees within the megablob, storing their tree clvs and tree probs
        std::vector<std::pair<double, std::vector<double>> > tree_clvs;
        tree_clvs.reserve(n_trees);

        setReticulationParents(blobInfo, megablob_idx, 0);
        for (size_t i = 0; i < n_trees; ++i) {
            size_t treeIdx = i;
            Node *startNode = nullptr;

            if (useGrayCode) {
                treeIdx = i ^ (i >> 1); // graycode iteration order
                if (i > 0) {
                    size_t lastI = i - 1;
                    size_t lastTreeIdx = lastI ^ (lastI >> 1);
                    size_t onlyChangedBit = treeIdx ^ lastTreeIdx;
                    size_t changedBitPos = log2(onlyChangedBit);
                    bool changedBitIsSet = treeIdx & onlyChangedBit;
                    startNode = blobInfo.reticulation_nodes_per_megablob[megablob_idx][changedBitPos];
                    startNode->getReticulationData()->setActiveParentToggle(changedBitIsSet);
                }
            } else {
                setReticulationParents(blobInfo, megablob_idx, treeIdx);
            }

            double tree_prob = displayed_tree_blob_prob(ann_network, megablob_idx, partitionIdx);
            if (tree_prob == 0.0 && !update_reticulation_probs) {
                continue;
            }
            std::vector<double> persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
            compute_tree_logl_blobs(ann_network, incremental, parent, fake_treeinfo, megablob_idx, partitionIdx,
                    &persite_logl, startNode);

            if (update_reticulation_probs) { // TODO: Only do this if we weren't at a leaf
                updateBestPersiteLoglikelihoodsBlobs(network, blobInfo, megablob_idx, treeIdx, numSites,
                        best_persite_logl_network, persite_logl);
            }
            if (megablobRootClvIdx == network.root->clv_index) { // we have reached the overall network root
                //std::cout << "tree_prob: " << tree_prob << "\n";
                for (size_t s = 0; s < numSites; ++s) {
                    persite_lh_network[s] += exp(persite_logl[s]) * tree_prob;
                }
            }
            if (n_trees > 1) {
                // extract the tree root clv vector and put it into tree_clvs together with its displayed tree probability
                std::vector<double> treeRootCLV;
                treeRootCLV.assign(fake_treeinfo.partitions[partitionIdx]->clv[megablobRootClvIdx],
                        fake_treeinfo.partitions[partitionIdx]->clv[megablobRootClvIdx] + clv_len);
                tree_clvs.emplace_back(std::make_pair(tree_prob, treeRootCLV));
            }
        }

        if (n_trees > 1) {
            // merge the tree clvs into the megablob root clv
            merge_tree_clvs(tree_clvs, fake_treeinfo.partitions[partitionIdx], megablobRootClvIdx);

            // std::cout << "loglikelihood we would get from the merged megablob root clv with index " << megablobRootClvIdx << ": ";
            Node *dbg_root = blobInfo.megablob_roots[megablob_idx];
            Node *dbg_back;
            if (dbg_root == network.root) {
                dbg_back = getTargetNode(network, dbg_root->getLink());
            } else {
                dbg_back = parent[dbg_root->clv_index];
            }
            double dbg_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partitionIdx],
                    dbg_root->clv_index, dbg_root->scaler_index, dbg_back->clv_index, dbg_back->scaler_index,
                    dbg_root->getLink()->edge_pmatrix_index, fake_treeinfo.param_indices[partitionIdx], nullptr);
            //std::cout << dbg_logl << "\n";
        }
    }
    return persite_lh_network;
}

std::vector<double> compute_persite_lh(AnnotatedNetwork &ann_network, unsigned int partitionIdx,
        const std::vector<Node*> &parent, bool unlinked_mode, bool update_reticulation_probs, unsigned int numSites,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, std::vector<double> *treewise_logl =
                nullptr, bool incremental = true) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool useGrayCode = ann_network.options.use_graycode;

    std::vector<double> persite_lh_network(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);

    // Iterate over all displayed trees
    unsigned int num_reticulations = network.num_reticulations();
    size_t n_trees = 1 << num_reticulations;

    /*if (ann_network.fake_treeinfo->clv_valid[partitionIdx][network.root->clv_index]) {
     n_trees = 1;
     }*/

    // iterate over all displayed trees, storing their tree clvs and tree probs
    std::vector<std::pair<double, std::vector<double>> > tree_clvs;
    tree_clvs.reserve(n_trees);
    unsigned int states_padded = fake_treeinfo.partitions[partitionIdx]->states_padded;
    unsigned int sites = fake_treeinfo.partitions[partitionIdx]->sites;
    unsigned int rate_cats = fake_treeinfo.partitions[partitionIdx]->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    setReticulationParents(network, 0);
    for (size_t i = 0; i < n_trees; ++i) {
        size_t treeIdx = i;
        Node *startNode = nullptr;
        if (useGrayCode) {
            treeIdx = i ^ (i >> 1); // graycode iteration order
            if (i > 0) {
                size_t lastI = i - 1;
                size_t lastTreeIdx = lastI ^ (lastI >> 1);
                size_t onlyChangedBit = treeIdx ^ lastTreeIdx;
                size_t changedBitPos = log2(onlyChangedBit);
                bool changedBitIsSet = treeIdx & onlyChangedBit;
                startNode = network.reticulation_nodes[changedBitPos];
                startNode->getReticulationData()->setActiveParentToggle(changedBitIsSet);
            }
        } else {
            setReticulationParents(network, treeIdx);
        }
        double tree_prob = displayed_tree_prob(ann_network, treeIdx, unlinked_mode ? 0 : partitionIdx);
        /*if (n_trees == 1) {
         std::cout << "repeat case\n";
         tree_prob = 1.0;
         }*/
        if (tree_prob == 0.0 && !update_reticulation_probs) {
            continue;
        }

        std::vector<double> persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
        double tree_logl = compute_tree_logl(ann_network, treeIdx, partitionIdx, &persite_logl, parent, startNode);
        //std::cout << "tree logl: " << tree_logl << "\n";
        if (treewise_logl) {
            (*treewise_logl)[treeIdx] = tree_logl;
        }
        for (size_t s = 0; s < numSites; ++s) {
            persite_lh_network[s] += exp(persite_logl[s]) * tree_prob;
        }

        if (n_trees > 1) {
            // extract the tree root clv vector and put it into tree_clvs together with its displayed tree probability
            std::vector<double> treeRootCLV;
            treeRootCLV.assign(fake_treeinfo.partitions[partitionIdx]->clv[network.root->clv_index],
                    fake_treeinfo.partitions[partitionIdx]->clv[network.root->clv_index] + clv_len);
            tree_clvs.emplace_back(std::make_pair(tree_prob, treeRootCLV));
        }

        if (update_reticulation_probs) {
            updateBestPersiteLoglikelihoods(treeIdx, network.num_reticulations(), numSites, best_persite_logl_network,
                    persite_logl);
        }
    }

    if (n_trees > 1) {
        // merge the tree clvs into the root clv
        merge_tree_clvs(tree_clvs, fake_treeinfo.partitions[partitionIdx], network.root->clv_index);
    }

    return persite_lh_network;
}

// TODO: Add bool incremental...
double processPartition(AnnotatedNetwork &ann_network, unsigned int partition_idx, int incremental,
        bool update_reticulation_probs, std::vector<unsigned int> &totalTaken, std::vector<unsigned int> &totalNotTaken,
        bool unlinked_mode, bool &reticulationProbsHaveChanged, std::vector<double> *treewise_logl) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool useBlobs = ann_network.options.use_blobs;
    bool useGrayCode = ann_network.options.use_graycode;
    bool useIncrementalClv = ann_network.options.use_incremental & incremental;

    unsigned int numSites = fake_treeinfo.partitions[partition_idx]->sites;
    std::vector<BestPersiteLoglikelihoodData> best_persite_logl_network;
    if (update_reticulation_probs) {
        best_persite_logl_network = std::vector<BestPersiteLoglikelihoodData>(numSites,
                BestPersiteLoglikelihoodData(network.num_reticulations()));
        reticulationProbsHaveChanged = false;
    }

    std::vector<Node*> parent;
    if (useBlobs || useGrayCode) {
        parent = grab_current_node_parents(network);
    }
    std::vector<double> persite_lh_network;

    if (!useBlobs) {
        persite_lh_network = compute_persite_lh(ann_network, partition_idx, parent, unlinked_mode,
                update_reticulation_probs, numSites, best_persite_logl_network, treewise_logl, useIncrementalClv);
    } else {
        if (treewise_logl) {
            throw std::runtime_error("Can't compute treewise logl with the blob optimization");
        }
        persite_lh_network = compute_persite_lh_blobs(ann_network, partition_idx, parent, unlinked_mode,
                update_reticulation_probs, numSites, best_persite_logl_network, useIncrementalClv);
    }

    double network_partition_logl = 0.0;
    for (size_t s = 0; s < numSites; ++s) {
        network_partition_logl += log(persite_lh_network[s]);
    }
    //std::cout << "network_partition_logl: " << network_partition_logl << "\n";

    fake_treeinfo.partition_loglh[partition_idx] = network_partition_logl;

    if (update_reticulation_probs) {
        update_total_taken(totalTaken, totalNotTaken, unlinked_mode, numSites, network.num_reticulations(),
                best_persite_logl_network);
        if (unlinked_mode) {
            reticulationProbsHaveChanged = update_probs(ann_network, partition_idx, totalTaken, totalNotTaken);
        }
    }

    // just for debug: print the persite lh
    /*std::cout << "persite likelihoods: \n";
     for (size_t i = 0; i < numSites; ++i) {
     std::cout << persite_lh_network[i] << "\n";
     }*/

    /*
     // just for debug: print the clv vectors
     for (size_t i = 0; i < network.num_nodes(); ++i) {
     printClv(fake_treeinfo, network.nodes[i].clv_index, partition_idx);
     }*/

    return network_partition_logl;
}

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices,
        bool update_reticulation_probs, std::vector<double> *treewise_logl) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;

    // special case: root node has a valid merged clv already
    if (ann_network.options.use_incremental & incremental & fake_treeinfo.clv_valid[0][network.root->clv_index]
            & !update_reticulation_probs) {
        return ann_network.old_logl;
    }
    setup_pmatrices(ann_network, incremental & ann_network.options.use_incremental, update_pmatrices);
    const int old_active_partition = fake_treeinfo.active_partition;
    fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    std::vector<unsigned int> totalTaken(network.num_reticulations(), 0);
    std::vector<unsigned int> totalNotTaken(network.num_reticulations(), 0);
    bool reticulationProbsHaveChanged = false;

    size_t n_trees = 1 << network.num_reticulations();

    double network_logl = 0;
    std::vector<double> treewise_partition_logl;
    if (treewise_logl) {
        for (size_t i = 0; i < n_trees; ++i) {
            treewise_logl->emplace_back(0.0);
        }
    }

    // Iterate over all partitions
    for (size_t partitionIdx = 0; partitionIdx < fake_treeinfo.partition_count; ++partitionIdx) {
        fake_treeinfo.active_partition = partitionIdx;

        std::vector<double> treewise_partition_logl;
        if (treewise_logl) {
            treewise_partition_logl = std::vector<double>(n_trees, 0.0);
        }

        double network_partition_logl;
        if (treewise_logl) {
            network_partition_logl = processPartition(ann_network, partitionIdx, incremental, update_reticulation_probs,
                    totalTaken, totalNotTaken, unlinked_mode, reticulationProbsHaveChanged, &treewise_partition_logl);
        } else {
            network_partition_logl = processPartition(ann_network, partitionIdx, incremental, update_reticulation_probs,
                    totalTaken, totalNotTaken, unlinked_mode, reticulationProbsHaveChanged, nullptr);
        }

        network_logl += network_partition_logl;
        if (treewise_logl) {
            for (size_t i = 0; i < treewise_partition_logl.size(); ++i) {
                (*treewise_logl)[i] += treewise_partition_logl[i];
            }
        }
    }

    if (update_reticulation_probs && !unlinked_mode) {
        reticulationProbsHaveChanged = update_probs(ann_network, 0, totalTaken, totalNotTaken);
    }

    /* restore original active partition */
    fake_treeinfo.active_partition = old_active_partition;

    if (update_reticulation_probs && reticulationProbsHaveChanged) {
        // invalidate clv entries
        for (size_t i = 0; i < network.num_reticulations(); ++i) {
            invalidateHigherCLVs(ann_network, network.reticulation_nodes[i]);
        }
        return computeLoglikelihood(ann_network, incremental, false, false);
    } else {
        if (ann_network.options.use_incremental) { // validate all clvs, for all partitions
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                for (size_t i = 0; i < network.num_nodes(); ++i) {
                    ann_network.fake_treeinfo->clv_valid[p][network.nodes[i].clv_index] = 1;
                }
            }
        }
        ann_network.old_logl = network_logl;
        return network_logl;
    }
}

double computeLoglikelihoodNaiveUtree(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices,
        std::vector<double> *treewise_logl) {
    (void) incremental;
    (void) update_pmatrices;

    Network &network = ann_network.network;
    RaxmlWrapper wrapper(ann_network.options);

    assert(wrapper.num_partitions() == 1);

    size_t n_trees = 1 << network.num_reticulations();
    double network_l = 0.0;
// Iterate over all displayed trees
    for (size_t i = 0; i < n_trees; ++i) {
        double tree_prob = displayed_tree_prob(ann_network, i, 0);
        if (tree_prob == 0.0) {
            continue;
        }

        pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, i);

        TreeInfo *displayedTreeinfo = wrapper.createRaxmlTreeinfo(displayed_tree);

        double tree_logl = displayedTreeinfo->loglh(0);

        delete displayedTreeinfo;

        if (treewise_logl) {
            treewise_logl->emplace_back(tree_logl);
        }

        assert(tree_logl != -std::numeric_limits<double>::infinity());
        network_l += exp(tree_logl) * tree_prob;
    }

    return log(network_l);
}

}
