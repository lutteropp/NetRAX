/*
 * NetworkIOTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"
#include "src/Api.hpp"
#include "src/graph/NetworkFunctions.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/graph/Common.hpp"
#include "old_likelihood/OldLikelihoodComputation.hpp"

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

TEST_F (LikelihoodTest, DISABLED_displayedTreeOfTreeToUtree) {
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

    for (size_t i = 0; i < treeNetwork.num_nodes(); ++i) {
        compareNodes(network_utree->nodes[i], raxml_utree->nodes[i]);
        compareNodes(network_utree->nodes[i]->back, raxml_utree->nodes[i]->back);
        if (network_utree->nodes[i]->next) {
            compareNodes(network_utree->nodes[i]->next, raxml_utree->nodes[i]->next);
            compareNodes(network_utree->nodes[i]->next->back, raxml_utree->nodes[i]->next->back);
            compareNodes(network_utree->nodes[i]->next->next, raxml_utree->nodes[i]->next->next);
            compareNodes(network_utree->nodes[i]->next->next->back,
                    raxml_utree->nodes[i]->next->next->back);

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
    pll_utree_traverse(utree->vroot, PLL_TREE_TRAVERSE_POSTORDER, cb_trav_all, outbuffer.data(),
            &trav_size);
    for (size_t i = 0; i < trav_size; ++i) {
        if (!outbuffer[i]->next) {
            labels.insert(outbuffer[i]->label);
        }
    }

    return labels;
}

void print_clv_index_by_label(const Network &network) {
    std::cout << "clv_index by node label:\n";
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        std::cout << network.nodes[i].label << ": " << network.nodes[i].clv_index << "\n";
    }
    std::cout << "\n";
}

bool no_clv_indices_equal(pll_utree_t *utree) {
    std::unordered_set<unsigned int> clv_idx;

    std::vector<pll_unode_t*> outbuffer(utree->inner_count + utree->tip_count);
    unsigned int trav_size;
    pll_utree_traverse(utree->vroot, PLL_TREE_TRAVERSE_POSTORDER, cb_trav_all, outbuffer.data(),
            &trav_size);
    for (size_t i = 0; i < trav_size; ++i) {
        clv_idx.insert(outbuffer[i]->clv_index);
    }
    return (clv_idx.size() == trav_size);
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

void compareLikelihoodFunctions(const std::string &networkPath, const std::string &msaPath,
        bool useRepeats) {
    NetraxOptions options;
    options.start_network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(options);
    Network &network = ann_network.network;
    print_clv_index_by_label(network);
    //std::cout << exportDebugInfo(network) << "\n";
    ASSERT_TRUE(networkIsConnected(network));

    //std::cout << exportDebugInfo(network) << "\n";
    ASSERT_TRUE(networkIsConnected(network));

    std::vector<double> treewise_logl_norep;
    std::vector<double> treewise_logl_naive;

    ann_network.options.use_blobs = false;
    ann_network.options.use_graycode = false;
    double norep_logl = old::computeLoglikelihood(ann_network, 0, 1, false, &treewise_logl_norep);
    ASSERT_NE(norep_logl, -std::numeric_limits<double>::infinity());
    ann_network.options.use_blobs = false;
    ann_network.options.use_graycode = true;
    double norep_logl_graycode = old::computeLoglikelihood(ann_network, 0, 1, false);
    ASSERT_NE(norep_logl, -std::numeric_limits<double>::infinity());
    ann_network.options.use_blobs = true;
    ann_network.options.use_graycode = false;
    double norep_logl_blobs = old::computeLoglikelihood(ann_network, 0, 1, false);
    ASSERT_NE(norep_logl_blobs, -std::numeric_limits<double>::infinity());
    ann_network.options.use_blobs = true;
    ann_network.options.use_graycode = true;
    double norep_logl_blobs_graycode = old::computeLoglikelihood(ann_network, 0, 1, false);
    ASSERT_NE(norep_logl_blobs_graycode, -std::numeric_limits<double>::infinity());
    double naive_logl = old::computeLoglikelihoodNaiveUtree(ann_network, 0, 1,
            &treewise_logl_naive);

    EXPECT_DOUBLE_EQ(norep_logl_graycode, norep_logl);
    EXPECT_DOUBLE_EQ(norep_logl_blobs, norep_logl);
    EXPECT_DOUBLE_EQ(norep_logl_blobs_graycode, norep_logl);
    if (naive_logl != -std::numeric_limits<double>::infinity()) {
        EXPECT_NEAR(naive_logl, norep_logl, 10);
    }

    EXPECT_EQ(treewise_logl_norep.size(), treewise_logl_naive.size());
    for (size_t i = 0; i < treewise_logl_norep.size(); ++i) {
        EXPECT_DOUBLE_EQ(treewise_logl_norep[i], treewise_logl_naive[i]);
    }

    std::cout << "norep_logl: " << norep_logl << "\n";
    std::cout << "norep_logl_graycode: " << norep_logl_graycode << "\n";
    std::cout << "norep_logl_blobs: " << norep_logl_blobs << "\n";
    std::cout << "norep_logl_blobs_graycode: " << norep_logl_blobs_graycode << "\n";
    std::cout << "naive logl: " << naive_logl << "\n";
}

void incrementalTest(const std::string &networkPath, const std::string &msaPath) {
    NetraxOptions options;
    options.start_network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = true;
    options.use_blobs = true;
    options.use_graycode = true;
    options.use_incremental = true;
    AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(options);
    Network &network = ann_network.network;
    print_clv_index_by_label(network);
    //std::cout << exportDebugInfo(network) << "\n";
    ASSERT_TRUE(networkIsConnected(network));

    double initial_logl = computeLoglikelihood(ann_network, 0, 1, false);
    ASSERT_NE(initial_logl, -std::numeric_limits<double>::infinity());

    double first_repeat = computeLoglikelihood(ann_network, 1, 0, false);
    ASSERT_NE(first_repeat, -std::numeric_limits<double>::infinity());

    EXPECT_DOUBLE_EQ(first_repeat, initial_logl);
}

TEST_F (LikelihoodTest, smallNetworkIncremental) {
    incrementalTest(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt");
}

TEST_F (LikelihoodTest, celineNetworkIncremental) {
    incrementalTest(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt");
}

TEST_F (LikelihoodTest, smallNetwork) {
    compareLikelihoodFunctions(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt",
            false);
}

TEST_F (LikelihoodTest, smallTree) {
    compareLikelihoodFunctions(DATA_PATH + "tree.nw", DATA_PATH + "small_fake_alignment.txt",
            false);
}

TEST_F (LikelihoodTest, smallTreeIncremental) {
    incrementalTest(DATA_PATH + "tree.nw", DATA_PATH + "small_fake_alignment.txt");
}

TEST_F (LikelihoodTest, tinyNetwork) {
    compareLikelihoodFunctions(DATA_PATH + "tiny.nw", DATA_PATH + "tiny_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, clvAveraging) {
    compareLikelihoodFunctions(DATA_PATH + "clv_averaging.nw",
            DATA_PATH + "5_taxa_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, twoReticulations) {
    compareLikelihoodFunctions(DATA_PATH + "two_reticulations.nw",
            DATA_PATH + "5_taxa_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, threeReticulations) {
    compareLikelihoodFunctions(DATA_PATH + "three_reticulations.nw",
            DATA_PATH + "7_taxa_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, interleavedReticulations) {
    compareLikelihoodFunctions(DATA_PATH + "interleaved_reticulations.nw",
            DATA_PATH + "5_taxa_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, reticulationInReticulation) {
    compareLikelihoodFunctions(DATA_PATH + "reticulation_in_reticulation.nw",
            DATA_PATH + "small_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, smallNetworkWithRepeats) {
    compareLikelihoodFunctions(DATA_PATH + "small.nw", DATA_PATH + "small_fake_alignment.txt",
            true);
}

TEST_F (LikelihoodTest, celineNetwork) {
    compareLikelihoodFunctions(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt",
            false);
}

TEST_F (LikelihoodTest, celineNetworkSmaller) {
    compareLikelihoodFunctions(DATA_PATH + "celine_smaller_1.nw",
            DATA_PATH + "celine_fake_alignment_smaller.txt", false);
}

TEST_F (LikelihoodTest, celineNetworkRepeats) {
    compareLikelihoodFunctions(DATA_PATH + "celine.nw", DATA_PATH + "celine_fake_alignment.txt",
            true);
}

TEST_F (LikelihoodTest, celineNetworkNonzeroBranches) {
    compareLikelihoodFunctions(DATA_PATH + "celine_nonzero_branches.nw",
            DATA_PATH + "celine_fake_alignment.txt", false);
}

TEST_F (LikelihoodTest, updateReticulationProb) {
    AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(
            NetraxOptions(networkPath, msaPath, false));
    double norep_logl = NetraxInstance::updateReticulationProbs(ann_network);
    std::cout << "norep_logl: " << norep_logl << "\n";
    double norep_logl_2 = NetraxInstance::updateReticulationProbs(ann_network);
    std::cout << "norep logl_2: " << norep_logl_2 << "\n";
    double norep_logl_3 = NetraxInstance::computeLoglikelihood(ann_network);
    std::cout << "norep logl_3: " << norep_logl_3 << "\n";

    EXPECT_EQ(norep_logl_2, norep_logl_3);
}

TEST_F (LikelihoodTest, simpleTreeWithRepeats) {
    AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(NetraxOptions(treePath, msaPath, true));
    double network_logl = NetraxInstance::computeLoglikelihood(ann_network);
    std::cout << "The computed network_logl 5 is: " << network_logl << "\n";
    EXPECT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, DISABLED_displayedTreeOfNetworkToUtree) {
    Network smallNetwork = netrax::readNetworkFromFile(networkPath);
    pll_utree_t *utree = displayed_tree_to_utree(smallNetwork, 0);
    EXPECT_NE(utree, nullptr);

    pll_utree_t *utree2 = displayed_tree_to_utree(smallNetwork, 0);
    EXPECT_NE(utree2, nullptr);

    // compare tip labels
    std::unordered_set<std::string> tip_labels_utree = collect_tip_labels_utree(utree);
    EXPECT_EQ(tip_labels_utree.size(), smallNetwork.num_tips());
    for (size_t i = 0; i < smallNetwork.num_tips(); ++i) {
        EXPECT_TRUE(tip_labels_utree.find(smallNetwork.nodes[i].label) != tip_labels_utree.end());
    }
    // compare tip labels for second tree
    std::unordered_set<std::string> tip_labels_utree2 = collect_tip_labels_utree(utree2);
    EXPECT_EQ(tip_labels_utree2.size(), smallNetwork.num_tips());
    for (size_t i = 0; i < smallNetwork.num_tips(); ++i) {
        EXPECT_TRUE(tip_labels_utree2.find(smallNetwork.nodes[i].label) != tip_labels_utree2.end());
    }

    // check for all different clvs
    EXPECT_TRUE(no_clv_indices_equal(utree));

    // check for all different clvs for second tree
    EXPECT_TRUE(no_clv_indices_equal(utree2));

    // TODO: check the branch lengths!!!
}

TEST_F (LikelihoodTest, buildAnnotatedNetworkTest) {
    NetraxOptions options;
    options.start_network_file = treePath;
    options.msa_file = msaPath;
    options.use_repeats = true;
    AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(options);
    ASSERT_TRUE(true);
}

TEST_F (LikelihoodTest, simpleTreeNaiveVersusNormalRaxml) {
    NetraxOptions options;
    options.start_network_file = treePath;
    options.msa_file = msaPath;
    options.use_repeats = true;
    AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(options);

    Network &network = ann_network.network;
    print_clv_index_by_label(network);

    double naive_logl = old::computeLoglikelihoodNaiveUtree(ann_network, 0, 1);

    pll_utree_t *raxml_utree = Tree::loadFromFile(treePath).pll_utree_copy();
    std::unique_ptr<RaxmlWrapper> treeWrapper = std::make_unique<RaxmlWrapper>(
            NetraxOptions(treePath, msaPath, false));
    TreeInfo *raxml_treeinfo = treeWrapper->createRaxmlTreeinfo(raxml_utree);
    double raxml_logl = raxml_treeinfo->loglh(false);

    delete raxml_treeinfo;

    EXPECT_NE(raxml_logl, -std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(raxml_logl, naive_logl);
}

TEST_F (LikelihoodTest, convertUtreeToNetwork) {
    NetraxOptions options;
    options.start_network_file = treePath;
    options.msa_file = msaPath;
    options.use_repeats = true;
    pll_utree_t *raxml_utree = Tree::loadFromFile(treePath).pll_utree_copy();

    AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network_from_utree(options, *raxml_utree);
    Network &network = ann_network.network;
    print_clv_index_by_label(network);
    double naive_utree_logl = old::computeLoglikelihoodNaiveUtree(ann_network, 0, 1);

    AnnotatedNetwork ann_network_2 = NetraxInstance::build_annotated_network(options);
    double naive_network_logl = old::computeLoglikelihoodNaiveUtree(ann_network_2, 0, 1);

    EXPECT_NE(naive_utree_logl, -std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(naive_network_logl, naive_utree_logl);
}
