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

double compute_tree_logl(AnnotatedNetwork &ann_network, std::vector<bool> &clv_touched,
        std::vector<bool> &dead_nodes, Node *displayed_tree_root, bool incremental,
        const std::vector<Node*> &parent, pllmod_treeinfo_t &fake_treeinfo, size_t megablob_idx,
        size_t partition_idx, std::vector<double> *persite_logl, Node *startNode = nullptr) {
    Network &network = ann_network.network;
    BlobInformation &blobInfo = ann_network.blobInfo;
    double tree_logl = 0.0;

    std::vector<pll_operation_t> ops;
    if (startNode) {
        ops = createOperationsUpdatedReticulation(ann_network, partition_idx, parent, startNode,
                dead_nodes, incremental, displayed_tree_root);
    } else {
        ops = createOperations(ann_network, partition_idx, parent, blobInfo, megablob_idx,
                dead_nodes, incremental, displayed_tree_root);
    }
    unsigned int ops_count = ops.size();
    if (ops_count == 0) {
        return 0.0;
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
    if (persite_logl != nullptr) {
        Node *rootBack = getTargetNode(network, ops_root->getLink());
        if (ops_root == network.root && !dead_nodes[rootBack->clv_index]) {
            tree_logl = pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partition_idx],
                    ops_root->clv_index, ops_root->scaler_index, rootBack->clv_index,
                    rootBack->scaler_index, ops_root->getLink()->edge_pmatrix_index,
                    fake_treeinfo.param_indices[partition_idx],
                    persite_logl->empty() ? nullptr : persite_logl->data());
        } else {
            tree_logl = pll_compute_root_loglikelihood(fake_treeinfo.partitions[partition_idx],
                    ops_root->clv_index, ops_root->scaler_index,
                    fake_treeinfo.param_indices[partition_idx],
                    persite_logl->empty() ? nullptr : persite_logl->data());
        }
    }
    return tree_logl;
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

void updateBestPersiteLoglikelihoods(Network &network, const BlobInformation &blobInfo,
        unsigned int megablob_idx, unsigned int treeIdx, unsigned int numSites,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network,
        const std::vector<double> &persite_logl) {
    for (size_t s = 0; s < numSites; ++s) {
        if (best_persite_logl_network[s].best_site_logl < persite_logl[s]) {
            std::fill(best_persite_logl_network[s].first_parent_taken_for_best_cnt.begin(),
                    best_persite_logl_network[s].first_parent_taken_for_best_cnt.end(), 0);
            best_persite_logl_network[s].best_site_logl = persite_logl[s];
        }
        size_t num_reticulations = blobInfo.reticulation_nodes_per_megablob[megablob_idx].size();
        for (size_t r = 0; r < num_reticulations; ++r) {
            if (!(treeIdx & (1 << r))) {
                continue;
            }
            size_t retIdxInNetwork = findReticulationIndexInNetwork(network,
                    blobInfo.reticulation_nodes_per_megablob[megablob_idx][r]);
            best_persite_logl_network[s].first_parent_taken_for_best_cnt[retIdxInNetwork]++;
        }
    }
}

void update_total_taken(std::vector<unsigned int> &totalTaken,
        std::vector<unsigned int> &totalNotTaken, bool unlinked_mode, unsigned int numSites,
        unsigned int num_reticulations,
        const std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network) {
    if (unlinked_mode) {
        std::fill(totalTaken.begin(), totalTaken.end(), 0);
        std::fill(totalNotTaken.begin(), totalNotTaken.end(), 0);
    }
    size_t n_trees = 1 << num_reticulations;
    for (size_t s = 0; s < numSites; ++s) {
        for (size_t r = 0; r < num_reticulations; ++r) {
            totalTaken[r] += best_persite_logl_network[s].first_parent_taken_for_best_cnt[r];
            totalNotTaken[r] += n_trees
                    - best_persite_logl_network[s].first_parent_taken_for_best_cnt[r];
        }
    }
}

bool update_probs(AnnotatedNetwork &ann_network, unsigned int partitionIdx,
        const std::vector<unsigned int> &totalTaken,
        const std::vector<unsigned int> &totalNotTaken) {
    Network &network = ann_network.network;
    bool reticulationProbsHaveChanged = false;
    for (size_t r = 0; r < network.num_reticulations(); ++r) {
        // Percentage of sites that were maximized when taking this reticulation
        double newProb = (double) totalTaken[r] / (totalTaken[r] + totalNotTaken[r]);

        size_t first_parent_pmatrix_index = getReticulationFirstParentPmatrixIndex(
                network.reticulation_nodes[r]);
        double oldProb = ann_network.branch_probs[partitionIdx][first_parent_pmatrix_index];

        if (newProb != oldProb) {
            size_t second_parent_pmatrix_index = getReticulationSecondParentPmatrixIndex(
                    network.reticulation_nodes[r]);
            ann_network.branch_probs[partitionIdx][first_parent_pmatrix_index] = newProb;
            ann_network.branch_probs[partitionIdx][second_parent_pmatrix_index] = 1.0 - newProb;
            reticulationProbsHaveChanged = true;
        }
    }
    return reticulationProbsHaveChanged;
}

void merge_tree_clvs(const std::vector<std::pair<double, std::vector<double>>> &tree_clvs,
        pll_partition_t *partition, unsigned int rootCLVIndex) {
    unsigned int states_padded = partition->states_padded;
    unsigned int sites = partition->sites;
    unsigned int rate_cats = partition->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    std::vector<mpfr::mpreal> clv_vec(clv_len, 0);

    for (unsigned int i = 0; i < clv_len; ++i) {
        clv_vec[i] = 0;
        for (unsigned int k = 0; k < tree_clvs.size(); ++k) {
            mpfr::mpreal a = tree_clvs[k].first;
            mpfr::mpreal b = tree_clvs[k].second[i];

            clv_vec[i] += a * b;
        }
    }

    double *clv = partition->clv[rootCLVIndex];
    for (unsigned int i = 0; i < clv_len; ++i) {
        clv[i] = clv_vec[i].toDouble();
    }
}

mpfr::mpreal displayed_tree_prob(AnnotatedNetwork &ann_network, size_t megablob_idx,
        size_t partition_index) {
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


std::vector<mpfr::mpreal> compute_persite_lh(AnnotatedNetwork &ann_network, unsigned int partitionIdx,
        const std::vector<Node*> &parent, bool update_reticulation_probs, unsigned int numSites,
        std::vector<bool> &clv_touched,
        std::vector<BestPersiteLoglikelihoodData> &best_persite_logl_network, bool incremental,
        std::vector<bool> *touched) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    BlobInformation &blobInfo = ann_network.blobInfo;

    unsigned int states_padded = fake_treeinfo.partitions[partitionIdx]->states_padded;
    unsigned int sites = fake_treeinfo.partitions[partitionIdx]->sites;
    unsigned int rate_cats = fake_treeinfo.partitions[partitionIdx]->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    std::vector<mpfr::mpreal> persite_lh_network(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
    std::vector<double> old_persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
    std::vector<bool> wasPruned(blobInfo.megablob_roots.size(), false);
    // Iterate over all megablobs in a bottom-up manner
    for (size_t megablob_idx = 0; megablob_idx < blobInfo.megablob_roots.size(); ++megablob_idx) {
        unsigned int megablobRootClvIdx = blobInfo.megablob_roots[megablob_idx]->clv_index;
        if (incremental && megablobRootClvIdx != network.root->clv_index
                && fake_treeinfo.clv_valid[partitionIdx][megablobRootClvIdx]
                && !update_reticulation_probs) {
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

            treeIdx = i ^ (i >> 1);
            if (i > 0) {
                size_t lastI = i - 1;
                size_t lastTreeIdx = lastI ^ (lastI >> 1);
                size_t onlyChangedBit = treeIdx ^ lastTreeIdx;
                size_t changedBitPos = log2(onlyChangedBit);
                bool changedBitIsSet = treeIdx & onlyChangedBit;
                startNode = blobInfo.reticulation_nodes_per_megablob[megablob_idx][changedBitPos];
                startNode->getReticulationData()->setActiveParentToggle(changedBitIsSet);
            }

            mpfr::mpreal tree_prob = displayed_tree_prob(ann_network, megablob_idx, partitionIdx);
            if (tree_prob == 0.0 && !update_reticulation_probs) {
                continue;
            }
            Node *displayed_tree_root = nullptr;
            std::vector<bool> dead_nodes = collect_dead_nodes(network, megablobRootClvIdx,
                    &displayed_tree_root);
            std::vector<double> persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);

            if (startNode && dead_nodes[startNode->clv_index]) {
                persite_logl = old_persite_logl;
            } else {
                if (startNode && !clv_touched[startNode->clv_index]) {
                    fill_untouched_clvs(ann_network, clv_touched, dead_nodes, partitionIdx,
                            startNode);
                    assert(clv_touched[startNode->clv_index]);
                }
                double tree_logl = compute_tree_logl(ann_network, clv_touched, dead_nodes, displayed_tree_root,
                        incremental, parent, fake_treeinfo, megablob_idx, partitionIdx,
                        &persite_logl, startNode);
                old_persite_logl = persite_logl;
            }

            bool zeros = std::all_of(persite_logl.begin(), persite_logl.end(), [](double d) {
                return d == 0.0;
            });
            if (zeros) {
                wasPruned[megablob_idx] = true;
            }

            if (update_reticulation_probs) { // TODO: Only do this if we weren't at a leaf
                updateBestPersiteLoglikelihoods(network, blobInfo, megablob_idx, treeIdx, numSites,
                        best_persite_logl_network, persite_logl);
            }
            if (megablobRootClvIdx == network.root->clv_index) {
                if (!wasPruned[megablob_idx]) {
                    assert(clv_touched[displayed_tree_root->clv_index]);
                    for (size_t s = 0; s < numSites; ++s) {
                        mpfr::mpreal site_logl = persite_logl[s];
                        persite_lh_network[s] += mpfr::exp(site_logl) * tree_prob;
                    }
                }
            }
            if (n_trees > 1 && megablobRootClvIdx != network.root->clv_index) {
                size_t clvIdx = displayed_tree_root->clv_index;
                assert(clv_touched[clvIdx]);
                std::vector<double> treeRootCLV;
                treeRootCLV.assign(fake_treeinfo.partitions[partitionIdx]->clv[clvIdx],
                        fake_treeinfo.partitions[partitionIdx]->clv[clvIdx] + clv_len);
                tree_clvs.emplace_back(std::make_pair(tree_prob, treeRootCLV));
            }
        }

        if (n_trees > 1 && megablobRootClvIdx != network.root->clv_index) {
            assert(clv_touched[megablobRootClvIdx]);
            merge_tree_clvs(tree_clvs, fake_treeinfo.partitions[partitionIdx], megablobRootClvIdx);
        }
    }

    if (wasPruned[blobInfo.megablob_roots.size() - 1]) {
        assert(
                blobInfo.megablob_roots[blobInfo.megablob_roots.size() - 1]->clv_index
                        == network.root->clv_index);
        bool foundUnpruned = false;
        unsigned int megablobRootClvIdx;
        for (int i = blobInfo.megablob_roots.size() - 2; i >= 0; --i) {
            if (!wasPruned[i]) {
                foundUnpruned = true;
                megablobRootClvIdx = blobInfo.megablob_roots[i]->clv_index;
                break;
            }
        }
        assert(megablobRootClvIdx < clv_touched.size());
        assert(clv_touched[megablobRootClvIdx]);
        std::vector<double> persite_logl(fake_treeinfo.partitions[partitionIdx]->sites, 0.0);
        pll_compute_edge_loglikelihood(fake_treeinfo.partitions[partitionIdx], network.nodes.size(),
                -1, megablobRootClvIdx, network.nodes_by_index[megablobRootClvIdx]->scaler_index,
                network.edges.size(), fake_treeinfo.param_indices[partitionIdx],
                persite_logl.data());
        assert(foundUnpruned);
        for (size_t s = 0; s < numSites; ++s) {
            mpfr::mpreal site_logl = persite_logl[s];
            persite_lh_network[s] += mpfr::exp(site_logl);
        }
    }

    if (touched) {
        *touched = clv_touched;
    }
    return persite_lh_network;
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

double processPartition(AnnotatedNetwork &ann_network, unsigned int partition_idx, int incremental,
        bool update_reticulation_probs, std::vector<unsigned int> &totalTaken,
        std::vector<unsigned int> &totalNotTaken, bool unlinked_mode,
        bool &reticulationProbsHaveChanged, std::vector<bool> *touched) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool useIncrementalClv = ann_network.options.use_incremental & incremental;

    unsigned int numSites = fake_treeinfo.partitions[partition_idx]->sites;
    std::vector<BestPersiteLoglikelihoodData> best_persite_logl_network;
    if (update_reticulation_probs) {
        best_persite_logl_network = std::vector<BestPersiteLoglikelihoodData>(numSites,
                BestPersiteLoglikelihoodData(network.num_reticulations()));
        reticulationProbsHaveChanged = false;
    }

    std::vector<Node*> parent = grab_current_node_parents(network);
    std::vector<mpfr::mpreal> persite_lh_network;
    std::vector<bool> clv_touched = init_clv_touched(ann_network, incremental, partition_idx);

    persite_lh_network = compute_persite_lh(ann_network, partition_idx, parent,
            update_reticulation_probs, numSites, clv_touched, best_persite_logl_network,
            useIncrementalClv, touched);

    mpfr::mpreal network_partition_logl = 0.0;
    for (size_t s = 0; s < numSites; ++s) {
        network_partition_logl += mpfr::log(persite_lh_network[s]);
    }

    fake_treeinfo.partition_loglh[partition_idx] = network_partition_logl.toDouble();

    if (update_reticulation_probs) {
        update_total_taken(totalTaken, totalNotTaken, unlinked_mode, numSites,
                network.num_reticulations(), best_persite_logl_network);
        if (unlinked_mode) {
            reticulationProbsHaveChanged = update_probs(ann_network, partition_idx, totalTaken,
                    totalNotTaken);
        }
    }

    return network_partition_logl.toDouble();
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

double computeLoglikelihood_broken(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices,
        bool update_reticulation_probs) {
    assert(ann_network.options.use_blobs & ann_network.options.use_graycode);
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    if (incremental & fake_treeinfo.clv_valid[0][network.root->clv_index]
            & !update_reticulation_probs) {
        return ann_network.old_logl;
    }
    double network_logl = 0;
    assert(!ann_network.branch_probs.empty());
    setup_pmatrices(ann_network, incremental, update_pmatrices);
    const int old_active_partition = fake_treeinfo.active_partition;
    fake_treeinfo.active_partition = PLLMOD_TREEINFO_PARTITION_ALL;
    bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
    std::vector<unsigned int> totalTaken(network.num_reticulations(), 0);
    std::vector<unsigned int> totalNotTaken(network.num_reticulations(), 0);
    bool reticulationProbsHaveChanged = false;
    std::vector<bool> touched(network.nodes.size(), false);

    for (size_t partitionIdx = 0; partitionIdx < fake_treeinfo.partition_count; ++partitionIdx) {
        fake_treeinfo.active_partition = partitionIdx;
        std::vector<bool> *touched_ptr = nullptr;
        if (partitionIdx == 0) {
            touched_ptr = &touched;
        }
        double network_partition_logl = processPartition(ann_network, partitionIdx, incremental,
                update_reticulation_probs, totalTaken, totalNotTaken, unlinked_mode,
                reticulationProbsHaveChanged, touched_ptr);
        network_logl += network_partition_logl;
    }
    if (update_reticulation_probs && !unlinked_mode) {
        reticulationProbsHaveChanged = update_probs(ann_network, 0, totalTaken, totalNotTaken);
    }
    /* restore original active partition */
    fake_treeinfo.active_partition = old_active_partition;

    assertReticulationProbs(ann_network);
    if (update_reticulation_probs && reticulationProbsHaveChanged) {
        std::vector<bool> visited(network.nodes.size(), false);
        for (size_t i = 0; i < network.num_reticulations(); ++i) {
            invalidateHigherCLVs(ann_network, network.reticulation_nodes[i], false, visited);
        }
        return computeLoglikelihood(ann_network, incremental, false, false);
    } else {
        // validate all touched clvs, for all partitions
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            for (size_t i = 0; i < network.num_nodes(); ++i) {
                ann_network.fake_treeinfo->clv_valid[p][network.nodes[i].clv_index] |=
                        touched[network.nodes[i].clv_index];
            }
        }
        if (network_logl >= -1) {
            std::cout << exportDebugInfo(network) << "\n";
        }
        assert(network_logl < -1);
        ann_network.old_logl = network_logl;
        return network_logl;
    }
}

double displayed_tree_logprob(AnnotatedNetwork &ann_network, size_t tree_index,
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
            std::cout << "The start node is a dead node!!!\n";
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
    std::vector<double> best_persite_logl(n_sites, std::numeric_limits<double>::min());
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
        // find the reticulation nodes that have taken their first parent in the best tree
        for (size_t j = 0; j < ann_network.network.num_reticulations(); ++j) {
            if (best_tree_idx[i] & (1 << j)) {
                total_taken[j]++;
            }
        }
    }

    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        double old_prob = ann_network.branch_probs[partition_idx][i];
        double new_prob = (double) total_taken[i] / n_sites;
        ann_network.branch_probs[partition_idx][i] = new_prob;
        changed |= (old_prob != new_prob);
    }

    return changed;
}

bool update_reticulation_probs_linked(AnnotatedNetwork &ann_network, const std::vector<std::vector<DisplayedTreeData>>& displayed_trees_all) {
    bool changed = false;
 

    throw std::runtime_error("Not implemented yet");
    return changed;
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

        mpfr::mpreal partition_lh = 0.0;
        for (const auto& tree : displayed_trees) {
            assert(tree.tree_logl != 0);
            partition_lh += mpfr::exp(tree.tree_logprob) * mpfr::exp(tree.tree_logl);
        }
        network_logl += mpfr::log(partition_lh);

        if (update_reticulation_probs) {
            if (unlinked_mode) {
                reticulation_probs_changed |= update_reticulation_probs_unlinked(ann_network, displayed_trees, partition_idx);
            } else {
                displayed_trees_all.emplace_back(displayed_trees);
            }
        }
    }
    if (update_reticulation_probs && !unlinked_mode) {
        reticulation_probs_changed = update_reticulation_probs_linked(ann_network, displayed_trees_all);
    }

    ann_network.old_logl = network_logl.toDouble();

    if (reticulation_probs_changed) {
        std::vector<bool> visited(network.nodes.size(), false);
        for (size_t i = 0; i < network.num_reticulations(); ++i) {
            invalidateHigherCLVs(ann_network, network.reticulation_nodes[i], false, visited);
        }
        return computeLoglikelihood_new(ann_network, incremental, false, false);
    }
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
