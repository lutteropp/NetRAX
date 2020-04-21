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

pll_operation_t buildOperationInternal(Node *parent, Node *child1, Node *child2, size_t fake_clv_index,
        size_t fake_pmatrix_index) {
    pll_operation_t operation;
    assert(parent);
    assert(child1 || child2);
    operation.parent_clv_index = parent->clv_index;
    operation.parent_scaler_index = parent->scaler_index;
    if (child1) {
        operation.child1_clv_index = child1->clv_index;
        operation.child1_scaler_index = child1->scaler_index;
        operation.child1_matrix_index = getEdgeTo(child1, parent)->pmatrix_index;
    } else {
        operation.child1_clv_index = fake_clv_index;
        operation.child1_scaler_index = -1;
        operation.child1_matrix_index = fake_pmatrix_index;
    }
    if (child2) {
        operation.child2_clv_index = child2->clv_index;
        operation.child2_scaler_index = child2->scaler_index;
        operation.child2_matrix_index = getEdgeTo(child2, parent)->pmatrix_index;
    } else {
        operation.child2_clv_index = fake_clv_index;
        operation.child2_scaler_index = -1;
        operation.child2_matrix_index = fake_pmatrix_index;
    }
    return operation;
}

pll_operation_t buildOperation(Node *actNode, Node *parentNode, const std::vector<bool> &dead_nodes,
        size_t fake_clv_index, size_t fake_pmatrix_index) {
    std::vector<Node*> activeChildren = getActiveChildren(actNode, parentNode);
    assert(activeChildren.size() > 0);
    Node *child1 = nullptr;
    if (!dead_nodes[activeChildren[0]->clv_index]) {
        child1 = activeChildren[0];
    }
    Node *child2 = nullptr;
    if (activeChildren.size() == 2 && !dead_nodes[activeChildren[1]->clv_index]) {
        child2 = activeChildren[1];
    }
    return buildOperationInternal(actNode, child1, child2, fake_clv_index, fake_pmatrix_index);
}

void createOperationsPostorder(Node *parent, Node *actNode, std::vector<pll_operation_t> &ops, size_t fake_clv_index,
        size_t fake_pmatrix_index, const std::vector<bool> &dead_nodes, const std::vector<unsigned int> *stop_indices =
                nullptr) {
    if (stop_indices
            && std::find(stop_indices->begin(), stop_indices->end(), actNode->clv_index) != stop_indices->end()) {
        return;
    }

    std::vector<Node*> activeChildren = getActiveChildren(actNode, parent);
    if (activeChildren.empty()) { // nothing to do if we are at a leaf node
        return;
    }
    assert(activeChildren.size() <= 2);
    for (size_t i = 0; i < activeChildren.size(); ++i) {
        if (!dead_nodes[activeChildren[i]->clv_index]) {
            createOperationsPostorder(actNode, activeChildren[i], ops, fake_clv_index, fake_pmatrix_index, dead_nodes,
                    stop_indices);
        }
    }

    pll_operation_t operation = buildOperation(actNode, parent, dead_nodes, fake_clv_index, fake_pmatrix_index);
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

std::vector<pll_operation_t> createOperations(Network &network, const std::vector<Node*> &parent,
        BlobInformation &blobInfo, unsigned int megablobIdx, const std::vector<bool> &dead_nodes) {
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
        createOperationsPostorder(network.root, getTargetNode(network.root->getLink()), ops, fake_clv_index,
                fake_pmatrix_index, dead_nodes, &stopIndices);
        createOperationsPostorder(getTargetNode(network.root->getLink()), network.root, ops, fake_clv_index,
                fake_pmatrix_index, dead_nodes, &stopIndices);
    } else {
        Node *megablobRoot = blobInfo.megablob_roots[megablobIdx];
        createOperationsPostorder(parent[megablobRoot->clv_index], megablobRoot, ops, fake_clv_index,
                fake_pmatrix_index, dead_nodes, &stopIndices);
    }
    //printOperationArray(ops);
    return ops;
}

Node* getActiveParent(Node *actNode, const std::vector<Node*> &parent) {
    Node *actParent;
    if (actNode->type == NodeType::RETICULATION_NODE) {
        actParent = getReticulationActiveParent(actNode);
    } else {
        actParent = parent[actNode->clv_index];
    }
    return actParent;
}

std::vector<pll_operation_t> createOperationsTowardsRoot(Network &network, const std::vector<Node*> &parent,
        Node *actParent, const std::vector<bool> &dead_nodes) {
    std::vector<pll_operation_t> ops;
    size_t fake_clv_index = network.nodes.size();
    size_t fake_pmatrix_index = network.edges.size();
    Node *rootBack = getTargetNode(network.root->getLink());

    while (actParent != network.root && actParent != rootBack) {
        ops.emplace_back(
                buildOperation(actParent, parent[actParent->clv_index], dead_nodes, fake_clv_index,
                        fake_pmatrix_index));
        actParent = getActiveParent(actParent, parent);
    }

    // now, add the two operations for the root node in reverse order.
    if (!rootBack->isTip()) {
        ops.push_back(buildOperation(rootBack, network.root, dead_nodes, fake_clv_index, fake_pmatrix_index));
    }
    ops.push_back(buildOperation(network.root, rootBack, dead_nodes, fake_clv_index, fake_pmatrix_index));

    assert(ops[ops.size() - 1].parent_clv_index == network.root->clv_index);
    //printOperationArray(ops);
    return ops;
}

std::vector<pll_operation_t> createOperationsUpdatedReticulation(Network &network, const std::vector<Node*> &parent,
        Node *actNode, const std::vector<bool> &dead_nodes) {
    std::vector<pll_operation_t> ops;

    Node *firstParent = getReticulationFirstParent(actNode);
    std::vector<pll_operation_t> opsFirst = createOperationsTowardsRoot(network, parent, firstParent, dead_nodes);
    Node *secondParent = getReticulationSecondParent(actNode);
    std::vector<pll_operation_t> opsSecond = createOperationsTowardsRoot(network, parent, secondParent, dead_nodes);

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

std::vector<pll_operation_t> createOperations(Network &network, size_t treeIdx) {
    std::vector<pll_operation_t> ops;
    size_t fake_clv_index = network.nodes.size();
    size_t fake_pmatrix_index = network.edges.size();
    setReticulationParents(network, treeIdx);

    std::vector<bool> dead_nodes(network.nodes.size(), false);
    fill_dead_nodes_recursive(nullptr, network.root, dead_nodes);

    Node *rootBack = getTargetNode(network.root->getLink());

    // How to do the operations at the top-level root trifurcation?
    // First with root->back, then with root...
    createOperationsPostorder(network.root, rootBack, ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
    createOperationsPostorder(rootBack, network.root, ops, fake_clv_index, fake_pmatrix_index, dead_nodes);
    //printOperationArray(ops);
    return ops;
}

double displayed_tree_prob(Network &network, size_t tree_index, size_t partition_index) {
    setReticulationParents(network, tree_index);
    double logProb = 0;
    for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
        logProb += log(network.reticulation_nodes[i]->getReticulationData()->getActiveProb(partition_index));
    }
    return exp(logProb);
}

double displayed_tree_prob(BlobInformation &blobInfo, size_t megablob_idx, size_t partition_index) {
    double logProb = 0;
    for (size_t i = 0; i < blobInfo.reticulation_nodes_per_megablob[megablob_idx].size(); ++i) {
        logProb += log(
                blobInfo.reticulation_nodes_per_megablob[megablob_idx][i]->getReticulationData()->getActiveProb(
                        partition_index));
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

double compute_tree_logl(Network &network, pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx, size_t partition_idx,
        std::vector<double> *persite_logl) {
// Create pll_operations_t array for the current displayed tree
    std::vector<pll_operation_t> ops = createOperations(network, tree_idx);
    unsigned int ops_count = ops.size();
    Node *ops_root = network.getNodeByClvIndex(ops[ops.size() - 1].parent_clv_index);
// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
    pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);

    Node *rootBack = getTargetNode(ops_root->getLink());

    double tree_partition_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx],
            ops_root->clv_index, ops_root->scaler_index, rootBack->clv_index, rootBack->scaler_index,
            ops_root->getLink()->edge->pmatrix_index, fake_treeinfo.param_indices[partition_idx],
            persite_logl->empty() ? nullptr : persite_logl->data());

    return tree_partition_logl;
}

void compute_tree_logl_blobs(Network &network, BlobInformation &blobInfo, const std::vector<Node*> &parent,
        pllmod_treeinfo_t &fake_treeinfo, size_t megablob_idx, size_t tree_idx, size_t partition_idx,
        std::vector<double> *persite_logl, Node *startNode = nullptr) {
    std::vector<bool> dead_nodes(network.nodes.size(), false);
    fill_dead_nodes_recursive(nullptr, network.root, dead_nodes);
// Create pll_operations_t array for the current displayed tree
    std::vector<pll_operation_t> ops;
    if (startNode) {
        ops = createOperationsUpdatedReticulation(network, parent, startNode, dead_nodes);
    } else {
        ops = createOperations(network, parent, blobInfo, megablob_idx, dead_nodes);
    }
    unsigned int ops_count = ops.size();
    if (ops_count == 0) {
        return;
    }
    Node *ops_root = network.getNodeByClvIndex(ops[ops.size() - 1].parent_clv_index);

// Compute CLVs in pll_update_partials, as specified by the operations array. This needs a pll_partition_t object.
    pll_update_partials(fake_treeinfo.partitions[partition_idx], ops.data(), ops_count);

    if (persite_logl != nullptr) {
        double tree_partition_logl;
        if (ops_root == network.root) {
            Node *rootBack = getTargetNode(ops_root->getLink());
            tree_partition_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx],
                    ops_root->clv_index, ops_root->scaler_index, rootBack->clv_index, rootBack->scaler_index,
                    ops_root->getLink()->edge->pmatrix_index, fake_treeinfo.param_indices[partition_idx],
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
void setup_pmatrices(Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental, int update_pmatrices) {
    /* NOTE: in unlinked brlen mode, up-to-date brlens for partition p
     * have to be prefetched to treeinfo->branch_lengths[p] !!! */
    bool collect_brlen = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? false : true);
    if (collect_brlen) {
        for (size_t i = 0; i < network.edges.size(); ++i) {
            fake_treeinfo.branch_lengths[0][network.edges[i].pmatrix_index] = network.edges[i].length;
        }
        // don't forget the fake entry
        fake_treeinfo.branch_lengths[0][network.edges.size()] = 0.0;
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
                    for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
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

bool update_probs(Network &network, unsigned int partitionIdx, const std::vector<unsigned int> &totalTaken,
        const std::vector<unsigned int> &totalNotTaken) {
    bool reticulationProbsHaveChanged = false;
    for (size_t r = 0; r < network.num_reticulations(); ++r) {
        double newProb = (double) totalTaken[r] / (totalTaken[r] + totalNotTaken[r]); // Percentage of sites that were maximized when taking this reticulation
        double oldProb = network.reticulation_nodes[r]->getReticulationData()->getProb(partitionIdx);
        if (newProb != oldProb) {
            reticulationProbsHaveChanged = true;
        }
        network.reticulation_nodes[r]->getReticulationData()->setProb(newProb, partitionIdx);
    }
    return reticulationProbsHaveChanged;
}

void merge_tree_clvs(const std::vector<std::pair<double, std::vector<double>>> &tree_clvs, pll_partition_t *partition,
        unsigned int rootCLVIndex) {
    unsigned int states = partition->states;
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

// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Add the blobs
std::vector<double> compute_persite_lh_blobs(unsigned int partitionIdx, Network &network, BlobInformation &blobInfo,
        const std::vector<Node*> &parent, pllmod_treeinfo_t &fake_treeinfo, bool unlinked_mode,
        bool update_reticulation_probs, unsigned int numSites,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, bool useGrayCode) {
    unsigned int states_padded = fake_treeinfo.partitions[partitionIdx]->states_padded;
    unsigned int sites = fake_treeinfo.partitions[partitionIdx]->sites;
    unsigned int rate_cats = fake_treeinfo.partitions[partitionIdx]->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    std::vector<double> persite_lh_network(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
    // Iterate over all megablobs in a bottom-up manner
    for (size_t megablob_idx = 0; megablob_idx < blobInfo.megablob_roots.size(); ++megablob_idx) {
        size_t n_trees = 1 << blobInfo.reticulation_nodes_per_megablob[megablob_idx].size();
        // iterate over all displayed trees within the megablob, storing their tree clvs and tree probs
        std::vector<std::pair<double, std::vector<double>> > tree_clvs;
        tree_clvs.reserve(n_trees);
        unsigned int megablobRootClvIdx = blobInfo.megablob_roots[megablob_idx]->clv_index;

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

            double tree_prob = displayed_tree_prob(blobInfo, megablob_idx, partitionIdx);
            if (tree_prob == 0.0 && !update_reticulation_probs) {
                continue;
            }
            std::vector<double> persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
            compute_tree_logl_blobs(network, blobInfo, parent, fake_treeinfo, megablob_idx, treeIdx,
                    partitionIdx, &persite_logl, startNode);

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
                dbg_back = getTargetNode(dbg_root->getLink());
            } else {
                dbg_back = parent[dbg_root->clv_index];
            }
            double dbg_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partitionIdx],
                    dbg_root->clv_index, dbg_root->scaler_index, dbg_back->clv_index, dbg_back->scaler_index,
                    dbg_root->getLink()->edge->pmatrix_index, fake_treeinfo.param_indices[partitionIdx], nullptr);
            //std::cout << dbg_logl << "\n";
        }
    }
    return persite_lh_network;
}

// TODO: Implement the Gray Code displayed tree iteration order and intelligent update of the operations array
// TODO: Add the blobs
std::vector<double> compute_persite_lh(unsigned int partitionIdx, Network &network, pllmod_treeinfo_t &fake_treeinfo,
        bool unlinked_mode, bool update_reticulation_probs, unsigned int numSites,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, std::vector<double> *treewise_logl =
                nullptr) {
    std::vector<double> persite_lh_network(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);

    // Iterate over all displayed trees
    unsigned int num_reticulations = network.num_reticulations();
    size_t n_trees = 1 << num_reticulations;
    for (size_t treeIdx = 0; treeIdx < n_trees; ++treeIdx) {
        double tree_prob = displayed_tree_prob(network, treeIdx, unlinked_mode ? 0 : partitionIdx);
        if (tree_prob == 0.0 && !update_reticulation_probs) {
            continue;
        }

        std::vector<double> persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
        double tree_logl = compute_tree_logl(network, fake_treeinfo, treeIdx, partitionIdx, &persite_logl);
        if (treewise_logl) {
            (*treewise_logl)[treeIdx] = tree_logl;
        }
        for (size_t s = 0; s < numSites; ++s) {
            persite_lh_network[s] += exp(persite_logl[s]) * tree_prob;
        }

        if (update_reticulation_probs) {
            updateBestPersiteLoglikelihoods(treeIdx, network.num_reticulations(), numSites, best_persite_logl_network,
                    persite_logl);
        }
    }

    return persite_lh_network;
}

// TODO: Add bool incremental...
double processPartition(unsigned int partitionIdx, Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental,
        bool update_reticulation_probs, std::vector<unsigned int> &totalTaken, std::vector<unsigned int> &totalNotTaken,
        bool unlinked_mode, bool &reticulationProbsHaveChanged, bool useBlobs, bool useGrayCode,
        std::vector<double> *treewise_logl) {
    unsigned int numSites = fake_treeinfo.partitions[partitionIdx]->sites;
    std::vector<BestPersiteLoglikelihoodData> best_persite_logl_network;
    if (update_reticulation_probs) {
        best_persite_logl_network = std::vector<BestPersiteLoglikelihoodData>(numSites,
                BestPersiteLoglikelihoodData(network.num_reticulations()));
        reticulationProbsHaveChanged = false;
    }

    std::vector<double> persite_lh_network;
    if (!useBlobs) {
        persite_lh_network = compute_persite_lh(partitionIdx, network, fake_treeinfo, unlinked_mode,
                update_reticulation_probs, numSites, best_persite_logl_network, treewise_logl);
    } else {
        if (treewise_logl) {
            throw std::runtime_error("Can't compute treewise logl with the blob optimization");
        }
        BlobInformation blobInfo = partitionNetworkIntoBlobs(network);
        std::vector<Node*> parent = grab_current_node_parents(network);
        persite_lh_network = compute_persite_lh_blobs(partitionIdx, network, blobInfo, parent, fake_treeinfo,
                unlinked_mode, update_reticulation_probs, numSites, best_persite_logl_network, useGrayCode);
    }

    double network_partition_logl = 0.0;
    for (size_t s = 0; s < numSites; ++s) {
        network_partition_logl += log(persite_lh_network[s]);
    }
    std::cout << "network_partition_logl: " << network_partition_logl << "\n";

    fake_treeinfo.partition_loglh[partitionIdx] = network_partition_logl;

    if (update_reticulation_probs) {
        update_total_taken(totalTaken, totalNotTaken, unlinked_mode, numSites, network.num_reticulations(),
                best_persite_logl_network);
        if (unlinked_mode) {
            reticulationProbsHaveChanged = update_probs(network, partitionIdx, totalTaken, totalNotTaken);
        }
    }

    return network_partition_logl;
}

double computeLoglikelihood(Network &network, pllmod_treeinfo_t &fake_treeinfo, int incremental, int update_pmatrices,
        bool update_reticulation_probs, bool useBlobs, bool useGrayCode, std::vector<double> *treewise_logl) {
    setup_pmatrices(network, fake_treeinfo, incremental, update_pmatrices);
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
            network_partition_logl = processPartition(partitionIdx, network, fake_treeinfo, incremental,
                            update_reticulation_probs, totalTaken, totalNotTaken, unlinked_mode, reticulationProbsHaveChanged,
                            useBlobs, useGrayCode, &treewise_partition_logl);
        } else {
            network_partition_logl = processPartition(partitionIdx, network, fake_treeinfo, incremental,
                            update_reticulation_probs, totalTaken, totalNotTaken, unlinked_mode, reticulationProbsHaveChanged,
                            useBlobs, useGrayCode, nullptr);
        }

        network_logl += network_partition_logl;
        if (treewise_logl) {
            for (size_t i = 0; i < treewise_partition_logl.size(); ++i) {
                (*treewise_logl)[i] += treewise_partition_logl[i];
            }
        }
    }

    if (update_reticulation_probs && !unlinked_mode) {
        reticulationProbsHaveChanged = update_probs(network, 0, totalTaken, totalNotTaken);
    }

    /* restore original active partition */
    fake_treeinfo.active_partition = old_active_partition;

    if (update_reticulation_probs && reticulationProbsHaveChanged) {
        return computeLoglikelihood(network, fake_treeinfo, incremental, false, false, useBlobs);
    } else {
        return network_logl;
    }
}

double computeLoglikelihoodNaiveUtree(RaxmlWrapper &wrapper, Network &network, int incremental, int update_pmatrices,
        std::vector<double> *treewise_logl) {
    (void) incremental;
    (void) update_pmatrices;

    assert(wrapper.num_partitions() == 1);

    size_t n_trees = 1 << network.reticulation_nodes.size();
    double network_l = 0.0;
// Iterate over all displayed trees
    for (size_t i = 0; i < n_trees; ++i) {
        double tree_prob = displayed_tree_prob(network, i, 0);
        if (tree_prob == 0.0) {
            continue;
        }

        pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, i);

        TreeInfo displayedTreeinfo = wrapper.createRaxmlTreeinfo(displayed_tree);

        double tree_logl = displayedTreeinfo.loglh(0);

        if (treewise_logl) {
            treewise_logl->emplace_back(tree_logl);
        }

        assert(tree_logl != -std::numeric_limits<double>::infinity());
        network_l += exp(tree_logl) * tree_prob;
    }

    return log(network_l);
}

}
