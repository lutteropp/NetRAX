/*
 * LikelihoodTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"

#include "src/graph/NetworkFunctions.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/graph/Common.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

const std::string DATA_PATH = "examples/sample_networks/";

std::mutex g_singleThread;

class LikelihoodTest: public ::testing::Test {
protected:
    std::string treePath = DATA_PATH + "tree.nw";
    std::string networkPath = DATA_PATH + "small.nw";
    std::string msaPath = DATA_PATH + "small_fake_alignment.txt";

    virtual void SetUp() {
        g_singleThread.lock();
    }

    virtual void TearDown() {
        g_singleThread.unlock();
    }
};

TEST_F (LikelihoodTest, testTheTest) {
    EXPECT_TRUE(true);
}

std::vector<size_t> getNeighborClvIndices(pll_unode_t *node) {
    std::vector<size_t> neighbors;
    if (node->next) {
        pll_unode_t *actNode = node;
        do {
            neighbors.push_back(actNode->back->clv_index);
            actNode = actNode->next;
        } while (actNode != node);
    }
    return neighbors;
}

void compareNodes(pll_unode_t *node1, pll_unode_t *node2) {
    EXPECT_EQ(node1->clv_index, node2->clv_index);
    // check if the clv indices of the neighbors are the same
    std::vector<size_t> node1Neighbors = getNeighborClvIndices(node1);
    std::vector<size_t> node2Neighbors = getNeighborClvIndices(node2);
    std::sort(node1Neighbors.begin(), node1Neighbors.end());
    std::sort(node2Neighbors.begin(), node2Neighbors.end());
    EXPECT_EQ(node1Neighbors.size(), node2Neighbors.size());
    for (size_t i = 0; i < node1Neighbors.size(); ++i) {
        EXPECT_EQ(node1Neighbors[i], node2Neighbors[i]);
    }

    EXPECT_EQ(node1->node_index, node2->node_index);
    EXPECT_EQ(node1->pmatrix_index, node2->pmatrix_index);
    EXPECT_EQ(node1->scaler_index, node2->scaler_index);
    EXPECT_EQ(node1->length, node2->length);
}

TEST_F (LikelihoodTest, displayedTreeOfTreeToUtree) {
    Network treeNetwork = netrax::readNetworkFromFile(treePath);
    pll_utree_t *network_utree = displayed_tree_to_utree(treeNetwork, 0);
    pll_utree_t *raxml_utree = Tree::loadFromFile(treePath).pll_utree_copy();

    EXPECT_NE(network_utree, nullptr);
    // compare the utrees:

    EXPECT_EQ(network_utree->inner_count, raxml_utree->inner_count);
    EXPECT_EQ(network_utree->binary, raxml_utree->binary);
    EXPECT_EQ(network_utree->edge_count, raxml_utree->edge_count);
    EXPECT_EQ(network_utree->tip_count, raxml_utree->tip_count);
    compareNodes(network_utree->vroot, raxml_utree->vroot);

    for (size_t i = 0; i < treeNetwork.nodes.size(); ++i) {
        compareNodes(network_utree->nodes[i], raxml_utree->nodes[i]);
        compareNodes(network_utree->nodes[i]->back, raxml_utree->nodes[i]->back);
        if (network_utree->nodes[i]->next) {
            compareNodes(network_utree->nodes[i]->next, raxml_utree->nodes[i]->next);
            compareNodes(network_utree->nodes[i]->next->back, raxml_utree->nodes[i]->next->back);
            compareNodes(network_utree->nodes[i]->next->next, raxml_utree->nodes[i]->next->next);
            compareNodes(network_utree->nodes[i]->next->next->back, raxml_utree->nodes[i]->next->next->back);

            compareNodes(network_utree->nodes[i]->next->next->next, network_utree->nodes[i]);
            compareNodes(raxml_utree->nodes[i]->next->next->next, raxml_utree->nodes[i]);
        }
    }
}

int cb_trav_all(pll_unode_t *node) {
    (void*) node;
    return 1;
}

std::unordered_set<std::string> collect_tip_labels_utree(pll_utree_t *utree) {
    std::unordered_set<std::string> labels;

    std::vector<pll_unode_t*> outbuffer(utree->inner_count + utree->tip_count);
    unsigned int trav_size;
    pll_utree_traverse(utree->vroot, PLL_TREE_TRAVERSE_POSTORDER, cb_trav_all, outbuffer.data(), &trav_size);
    for (size_t i = 0; i < trav_size; ++i) {
        if (!outbuffer[i]->next) {
            labels.insert(outbuffer[i]->label);
        }
    }

    return labels;
}

void print_clv_index_by_label(const Network &network) {
    std::cout << "clv_index by node label:\n";
    for (size_t i = 0; i < network.nodes.size(); ++i) {
        std::cout << network.nodes[i].label << ": " << network.nodes[i].clv_index << "\n";
    }
    std::cout << "\n";
}

bool no_clv_indices_equal(pll_utree_t *utree) {
    std::unordered_set<unsigned int> clv_idx;

    std::vector<pll_unode_t*> outbuffer(utree->inner_count + utree->tip_count);
    unsigned int trav_size;
    pll_utree_traverse(utree->vroot, PLL_TREE_TRAVERSE_POSTORDER, cb_trav_all, outbuffer.data(), &trav_size);
    for (size_t i = 0; i < trav_size; ++i) {
        clv_idx.insert(outbuffer[i]->clv_index);
    }
    return (clv_idx.size() == trav_size);
}

TEST_F (LikelihoodTest, displayedTreeOfNetworkToUtree) {
    Network smallNetwork = netrax::readNetworkFromFile(networkPath);
    pll_utree_t *utree = displayed_tree_to_utree(smallNetwork, 0);
    EXPECT_NE(utree, nullptr);

    pll_utree_t *utree2 = displayed_tree_to_utree(smallNetwork, 0);
    EXPECT_NE(utree2, nullptr);

    // compare tip labels
    std::unordered_set<std::string> tip_labels_utree = collect_tip_labels_utree(utree);
    EXPECT_EQ(tip_labels_utree.size(), smallNetwork.num_tips());
    for (size_t i = 0; i < smallNetwork.tip_nodes.size(); ++i) {
        EXPECT_TRUE(tip_labels_utree.find(smallNetwork.tip_nodes[i]->label) != tip_labels_utree.end());
    }
    // compare tip labels for second tree
    std::unordered_set<std::string> tip_labels_utree2 = collect_tip_labels_utree(utree2);
    EXPECT_EQ(tip_labels_utree2.size(), smallNetwork.num_tips());
    for (size_t i = 0; i < smallNetwork.tip_nodes.size(); ++i) {
        EXPECT_TRUE(tip_labels_utree2.find(smallNetwork.tip_nodes[i]->label) != tip_labels_utree2.end());
    }

    // check for all different clvs
    EXPECT_TRUE(no_clv_indices_equal(utree));

    // check for all different clvs for second tree
    EXPECT_TRUE(no_clv_indices_equal(utree2));

    // TODO: check the branch lengths!!!
}

TEST_F (LikelihoodTest, simpleTreeNoRepeatsNormalRaxml) {
    pll_utree_t *raxml_utree = Tree::loadFromFile(treePath).pll_utree_copy();
    std::unique_ptr<RaxmlWrapper> treeWrapper = std::make_unique<RaxmlWrapper>(NetraxOptions(treePath, msaPath, false));

    TreeInfo raxml_treeinfo = treeWrapper->createRaxmlTreeinfo(raxml_utree);

    double network_logl = raxml_treeinfo.loglh(false);
    std::cout << "The computed network_logl 1 is: " << network_logl << "\n";
    EXPECT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

void compare_clv(double *clv_raxml, double *clv_network, size_t clv_size) {
    for (size_t i = 0; i < clv_size; ++i) {
        EXPECT_EQ(clv_raxml[i], clv_network[i]);
    }
}

void comparePartitions(const pll_partition_t *p_network, const pll_partition_t *p_raxml) {
    EXPECT_EQ(p_network->tips, p_raxml->tips);
    EXPECT_EQ(p_network->clv_buffers, p_raxml->clv_buffers + 1);
    EXPECT_EQ(p_network->nodes, p_raxml->nodes + 1);
    EXPECT_EQ(p_network->states, p_raxml->states);
    EXPECT_EQ(p_network->sites, p_raxml->sites);
    EXPECT_EQ(p_network->pattern_weight_sum, p_raxml->pattern_weight_sum);
    EXPECT_EQ(p_network->rate_matrices, p_raxml->rate_matrices);
    EXPECT_EQ(p_network->prob_matrices, p_raxml->prob_matrices + 1);
    EXPECT_EQ(p_network->rate_cats, p_raxml->rate_cats);
    EXPECT_EQ(p_network->scale_buffers, p_raxml->scale_buffers + 1);
    EXPECT_EQ(p_network->attributes, p_raxml->attributes);
    EXPECT_EQ(p_network->alignment, p_raxml->alignment);
    EXPECT_EQ(p_network->states_padded, p_raxml->states_padded);

    // compare the clv vector entries...
    unsigned int start = (p_raxml->attributes & PLL_ATTRIB_PATTERN_TIP) ? p_raxml->tips : 0;
    unsigned int end = p_raxml->tips + p_raxml->clv_buffers;

    size_t sites_alloc = (unsigned int) p_raxml->asc_additional_sites + p_raxml->sites;

    size_t clv_size = sites_alloc * p_raxml->states_padded * p_raxml->rate_cats;
    for (size_t i = start; i < end; ++i) {
        compare_clv(p_raxml->clv[i], p_network->clv[i], clv_size);
    }
}

TEST_F (LikelihoodTest, comparePllmodTreeinfo) {
    Network treeNetwork = netrax::readNetworkFromFile(treePath);
    pll_utree_t *raxml_utree = Tree::loadFromFile(treePath).pll_utree_copy();
    std::unique_ptr<RaxmlWrapper> treeWrapper = std::make_unique<RaxmlWrapper>(NetraxOptions(treePath, msaPath, false));

    TreeInfo network_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(treeNetwork);
    TreeInfo raxml_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(raxml_utree);

    const pllmod_treeinfo_t &network_treeinfo = network_treeinfo_tree.pll_treeinfo();
    const pllmod_treeinfo_t &raxml_treeinfo = raxml_treeinfo_tree.pll_treeinfo();

    EXPECT_EQ(network_treeinfo.active_partition, raxml_treeinfo.active_partition);
    EXPECT_EQ(network_treeinfo.brlen_linkage, raxml_treeinfo.brlen_linkage);
    EXPECT_EQ(network_treeinfo.init_partition_count, raxml_treeinfo.init_partition_count);
    EXPECT_EQ(network_treeinfo.partition_count, raxml_treeinfo.partition_count);
    EXPECT_EQ(network_treeinfo.subnode_count, raxml_treeinfo.subnode_count + 3);
    EXPECT_EQ(network_treeinfo.tip_count, raxml_treeinfo.tip_count);

    for (size_t i = 0; i < raxml_treeinfo.partition_count; ++i) {
        comparePartitions(network_treeinfo.partitions[i], raxml_treeinfo.partitions[i]);
    }

    double network_logl = network_treeinfo_tree.loglh();
    double raxml_logl = raxml_treeinfo_tree.loglh();

    for (size_t i = 0; i < raxml_treeinfo.partition_count; ++i) {
        comparePartitions(network_treeinfo.partitions[i], raxml_treeinfo.partitions[i]);
    }

    EXPECT_EQ(network_logl, raxml_logl);
}

pll_unode_t* getNodeWithClvIndex(unsigned int clv_index, const pll_utree_t *tree) {
    for (size_t i = 0; i < tree->tip_count + tree->inner_count; ++i) {
        if (tree->nodes[i]->clv_index == clv_index) {
            return tree->nodes[i];
        }
    }
    throw std::runtime_error("There is no node with the given clv index");
}

bool isLeafNode(const pll_unode_t *node) {
    return (node->next == NULL);
}

TEST_F (LikelihoodTest, simpleNetworkNoRepeatsOnlyDisplayedTreeWithRaxml) {
    Network smallNetwork = netrax::readNetworkFromFile(networkPath);
    std::unique_ptr<RaxmlWrapper> treeWrapper = std::make_unique<RaxmlWrapper>(NetraxOptions(treePath, msaPath, false));
    TreeInfo raxml_treeinfo = treeWrapper->createRaxmlTreeinfo(displayed_tree_to_utree(smallNetwork, 0));

    double network_logl = raxml_treeinfo.loglh(false);
    std::cout << "The computed network_logl 3 is: " << network_logl << "\n";
    EXPECT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleNetworkWithRepeatsOnlyDisplayedTreeWithRaxml) {
    Network smallNetwork = netrax::readNetworkFromFile(networkPath);
    std::unique_ptr<RaxmlWrapper> treeWrapperRepeats = std::make_unique<RaxmlWrapper>(
            NetraxOptions(treePath, msaPath, true));

    TreeInfo raxml_treeinfo = treeWrapperRepeats->createRaxmlTreeinfo(displayed_tree_to_utree(smallNetwork, 0));

    double network_logl = raxml_treeinfo.loglh(false);
    std::cout << "The computed network_logl 4 is: " << network_logl << "\n";
    EXPECT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleTreeNoRepeats) {
    Network treeNetwork = netrax::readNetworkFromFile(treePath);
    std::unique_ptr<RaxmlWrapper> treeWrapper = std::make_unique<RaxmlWrapper>(NetraxOptions(treePath, msaPath, false));
    TreeInfo network_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(treeNetwork);
    double network_logl = network_treeinfo_tree.loglh(false);
    std::cout << "The computed network_logl 2 is: " << network_logl << "\n";
    EXPECT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, tree) {
    Network treeNetwork = netrax::readNetworkFromFile(treePath);
    pll_utree_t *utree = displayed_tree_to_utree(treeNetwork, 0);
    std::unique_ptr<RaxmlWrapper> treeWrapper = std::make_unique<RaxmlWrapper>(NetraxOptions(treePath, msaPath, false));
    TreeInfo raxml_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(utree);
    double raxml_logl = raxml_treeinfo_tree.loglh(0);
    std::cout << "raxml logl: " << raxml_logl << "\n";

    TreeInfo network_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(treeNetwork);
    double naive_logl = computeLoglikelihoodNaiveUtree(*(treeWrapper.get()), treeNetwork, 0, 1);
    std::cout << "naive logl: " << naive_logl << "\n";

    RaxmlWrapper::NetworkParams *params =
            (RaxmlWrapper::NetworkParams*) network_treeinfo_tree.pll_treeinfo().likelihood_computation_params;

    double norep_logl = computeLoglikelihood(treeNetwork, *(params->network_treeinfo), 0, 1);
    std::cout << "norep_logl: " << norep_logl << "\n";

    double norep_logl_blobs = computeLoglikelihood(treeNetwork, *(params->network_treeinfo), 0, 1, false, true);
    std::cout << "norep_logl_blobs: " << norep_logl_blobs << "\n";

    EXPECT_EQ(raxml_logl, naive_logl);
    EXPECT_EQ(naive_logl, norep_logl);
    EXPECT_EQ(norep_logl_blobs, norep_logl);
}

void compareLikelihoodFunctions(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    Network network = netrax::readNetworkFromFile(networkPath);
    print_clv_index_by_label(network);
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    RaxmlWrapper wrapper(options);
    std::cout << exportDebugInfo(network) << "\n";
    ASSERT_TRUE(networkIsConnected(network));
    TreeInfo network_treeinfo = wrapper.createRaxmlTreeinfo(network);
    std::cout << exportDebugInfo(network) << "\n";
    ASSERT_TRUE(networkIsConnected(network));
    RaxmlWrapper::NetworkParams *params =
            (RaxmlWrapper::NetworkParams*) network_treeinfo.pll_treeinfo().likelihood_computation_params;
    pllmod_treeinfo_t treeinfo = *(params->network_treeinfo);

    double norep_logl = computeLoglikelihood(network, treeinfo, 0, 1);
    ASSERT_NE(norep_logl, -std::numeric_limits<double>::infinity());
    std::cout << "norep_logl: " << norep_logl << "\n";
    double norep_logl_blobs = computeLoglikelihood(network, treeinfo, 0, 1, false, true);
    std::cout << "norep_logl_blobs: " << norep_logl_blobs << "\n";
    double naive_logl = computeLoglikelihoodNaiveUtree(wrapper, network, 0, 1);
    std::cout << "naive logl: " << naive_logl << "\n";

    if (naive_logl != -std::numeric_limits<double>::infinity()) {
        EXPECT_NEAR(naive_logl, norep_logl, 10);
    }

    EXPECT_NE(norep_logl, -std::numeric_limits<double>::infinity());
    EXPECT_NEAR(norep_logl_blobs, norep_logl, 1);
}

TEST_F (LikelihoodTest, smallNetwork) {
    compareLikelihoodFunctions(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, tinyNetwork) {
    compareLikelihoodFunctions(DATA_PATH + "tiny.nw", DATA_PATH + "tiny_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, clvAveraging) {
    compareLikelihoodFunctions(DATA_PATH + "clv_averaging.nw", DATA_PATH + "5_taxa_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, twoReticulations) {
    compareLikelihoodFunctions(DATA_PATH + "two_reticulations.nw", DATA_PATH + "5_taxa_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, threeReticulations) {
    compareLikelihoodFunctions(DATA_PATH + "three_reticulations.nw", DATA_PATH + "7_taxa_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, interleavedReticulations) {
    compareLikelihoodFunctions(DATA_PATH + "interleaved_reticulations.nw", DATA_PATH + "5_taxa_fake_alignment.txt",
            false);
}

TEST_F (LikelihoodTest, reticulationInReticulation) {
    compareLikelihoodFunctions(DATA_PATH + "reticulation_in_reticulation.nw", DATA_PATH + "small_fake_alignment.txt",
            false);
}

TEST_F (LikelihoodTest, smallNetworkWithRepeats) {
    compareLikelihoodFunctions(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt", true);
}

TEST_F (LikelihoodTest, celineNetwork) {
    compareLikelihoodFunctions(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, celineNetworkSmaller) {
    compareLikelihoodFunctions(DATA_PATH + "celine_smaller_1.nw", DATA_PATH + "celine_fake_alignment_smaller.txt",
            false);
}

TEST_F (LikelihoodTest, celineNetworkRepeats) {
    compareLikelihoodFunctions(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt", true);
}

TEST_F (LikelihoodTest, celineNetworkNonzeroBranches) {
    compareLikelihoodFunctions(DATA_PATH + "celine_nonzero_branches.nw", DATA_PATH + "celine_fake_alignment.txt",
            false);
}

TEST_F (LikelihoodTest, updateReticulationProb) {
    Network smallNetwork = netrax::readNetworkFromFile(networkPath);
    std::unique_ptr<RaxmlWrapper> smallWrapper = std::make_unique<RaxmlWrapper>(
            NetraxOptions(networkPath, msaPath, false));
    TreeInfo network_treeinfo_small = smallWrapper->createRaxmlTreeinfo(smallNetwork);
    RaxmlWrapper::NetworkParams *params =
            (RaxmlWrapper::NetworkParams*) network_treeinfo_small.pll_treeinfo().likelihood_computation_params;

    double norep_logl = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, true);
    std::cout << "norep_logl: " << norep_logl << "\n";
    double norep_logl_2 = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, true);
    std::cout << "norep logl_2: " << norep_logl_2 << "\n";
    double norep_logl_3 = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, false);
    std::cout << "norep logl_3: " << norep_logl_3 << "\n";

    EXPECT_EQ(norep_logl_2, norep_logl_3);
}

TEST_F (LikelihoodTest, simpleTreeWithRepeats) {
    Network treeNetwork = netrax::readNetworkFromFile(treePath);
    std::unique_ptr<RaxmlWrapper> treeWrapperRepeats = std::make_unique<RaxmlWrapper>(
            NetraxOptions(treePath, msaPath, true));
    TreeInfo network_treeinfo_tree = treeWrapperRepeats->createRaxmlTreeinfo(treeNetwork);
    double network_logl = network_treeinfo_tree.loglh(false);
    std::cout << "The computed network_logl 5 is: " << network_logl << "\n";
    EXPECT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

