/*
 * NetworkIOTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/Fake.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/Network.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

std::mutex g_singleThread;

class LikelihoodTest: public ::testing::Test {
protected:
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string networkPath = "examples/sample_networks/small.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance treeInstance;
	RaxmlInstance smallInstance;

	Network treeNetwork;
	Network smallNetwork;

	virtual void SetUp() {
		g_singleThread.lock();
		treeInstance = createStandardRaxmlInstance(treePath, msaPath, false);
		smallInstance = createStandardRaxmlInstance(networkPath, msaPath, false);
		treeNetwork = readNetworkFromFile(treePath);
		smallNetwork = readNetworkFromFile(networkPath);
	}

	virtual void TearDown() {
		g_singleThread.unlock();
	}
};

TEST_F (LikelihoodTest, testTheTest) {
	ASSERT_TRUE(true);
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
	ASSERT_EQ(node1->clv_index, node2->clv_index);
	// check if the clv indices of the neighbors are the same
	std::vector<size_t> node1Neighbors = getNeighborClvIndices(node1);
	std::vector<size_t> node2Neighbors = getNeighborClvIndices(node2);
	std::sort(node1Neighbors.begin(), node1Neighbors.end());
	std::sort(node2Neighbors.begin(), node2Neighbors.end());
	ASSERT_EQ(node1Neighbors.size(), node2Neighbors.size());
	for (size_t i = 0; i < node1Neighbors.size(); ++i) {
		ASSERT_EQ(node1Neighbors[i], node2Neighbors[i]);
	}

	ASSERT_EQ(node1->node_index, node2->node_index);
	ASSERT_EQ(node1->pmatrix_index, node2->pmatrix_index);
	ASSERT_EQ(node1->scaler_index, node2->scaler_index);
	ASSERT_EQ(node1->length, node2->length);
}

TEST_F (LikelihoodTest, displayedTreeOfTreeToUtree) {
	pll_utree_t *network_utree = displayed_tree_to_utree(treeNetwork, 0);

	ASSERT_NE(network_utree, nullptr);
	// compare the utrees:

	TreeInfo raxml_treeinfo_tree = createStandardRaxmlTreeinfo(treeInstance);

	pll_utree_t *raxml_utree = raxml_treeinfo_tree.pll_treeinfo().tree;
	ASSERT_EQ(network_utree->inner_count, raxml_utree->inner_count);
	ASSERT_EQ(network_utree->binary, raxml_utree->binary);
	ASSERT_EQ(network_utree->edge_count, raxml_utree->edge_count);
	ASSERT_EQ(network_utree->tip_count, raxml_utree->tip_count);
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

TEST_F (LikelihoodTest, displayedTreeOfNetworkToUtree) {
	pll_utree_t *utree = displayed_tree_to_utree(smallNetwork, 0);
	ASSERT_NE(utree, nullptr);
}

TEST_F (LikelihoodTest, simpleTreeNoRepeatsNormalRaxml) {
	TreeInfo raxml_treeinfo = createStandardRaxmlTreeinfo(treeInstance, false);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 1 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

void comparePartitions(const pll_partition_t *p1, const pll_partition_t *p2) {
	ASSERT_EQ(p1->tips, p2->tips);
	ASSERT_EQ(p1->clv_buffers, p2->clv_buffers);
	ASSERT_EQ(p1->nodes, p2->nodes);
	ASSERT_EQ(p1->states, p2->states);
	ASSERT_EQ(p1->sites, p2->sites);
	ASSERT_EQ(p1->pattern_weight_sum, p2->pattern_weight_sum);
	ASSERT_EQ(p1->rate_matrices, p2->rate_matrices);
	ASSERT_EQ(p1->prob_matrices, p2->prob_matrices);
	ASSERT_EQ(p1->rate_cats, p2->rate_cats);
	ASSERT_EQ(p1->scale_buffers, p2->scale_buffers);
	ASSERT_EQ(p1->attributes, p2->attributes);
	ASSERT_EQ(p1->alignment, p2->alignment);
	ASSERT_EQ(p1->states_padded, p2->states_padded);
}

TEST_F (LikelihoodTest, comparePllmodTreeinfo) {
	TreeInfo network_treeinfo_tree = createFakeRaxmlTreeinfo(treeInstance, treeNetwork);
	TreeInfo raxml_treeinfo_tree = createStandardRaxmlTreeinfo(treeInstance);

	const pllmod_treeinfo_t &network_treeinfo = network_treeinfo_tree.pll_treeinfo();
	const pllmod_treeinfo_t &raxml_treeinfo = raxml_treeinfo_tree.pll_treeinfo();

	ASSERT_EQ(network_treeinfo.active_partition, raxml_treeinfo.active_partition);
	ASSERT_EQ(network_treeinfo.brlen_linkage, raxml_treeinfo.brlen_linkage);
	ASSERT_EQ(network_treeinfo.init_partition_count, raxml_treeinfo.init_partition_count);
	ASSERT_EQ(network_treeinfo.partition_count, raxml_treeinfo.partition_count);
	ASSERT_EQ(network_treeinfo.subnode_count, raxml_treeinfo.subnode_count);
	ASSERT_EQ(network_treeinfo.tip_count, raxml_treeinfo.tip_count);

	for (size_t i = 0; i < raxml_treeinfo.partition_count; ++i) {
		comparePartitions(network_treeinfo.partitions[i], raxml_treeinfo.partitions[i]);
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

TEST_F (LikelihoodTest, compareOperationArrays) {
	pll_utree_t *network_utree = displayed_tree_to_utree(treeNetwork, 0);
	TreeInfo raxml_treeinfo_tree = createStandardRaxmlTreeinfo(treeInstance);
	raxml_treeinfo_tree.loglh(false); // to fill the operations array

	std::vector<pll_operation_t> network_ops = createOperations(treeNetwork, 0);
	pll_operation_t *raxml_ops = raxml_treeinfo_tree.pll_treeinfo().operations;

	pll_utree_t *raxml_utree = raxml_treeinfo_tree.pll_treeinfo().tree;

	std::cout << "Number of operations: " << network_ops.size() << "\n";
	for (size_t i = 0; i < network_ops.size(); ++i) {
		pll_unode_t *parent_network = getNodeWithClvIndex(network_ops[i].parent_clv_index, network_utree);
		pll_unode_t *child1_network = getNodeWithClvIndex(network_ops[i].child1_clv_index, network_utree);
		pll_unode_t *child2_network = getNodeWithClvIndex(network_ops[i].child2_clv_index, network_utree);
		pll_unode_t *parent_raxml = getNodeWithClvIndex(raxml_ops[i].parent_clv_index, raxml_utree);
		pll_unode_t *child1_raxml = getNodeWithClvIndex(raxml_ops[i].child1_clv_index, raxml_utree);
		pll_unode_t *child2_raxml = getNodeWithClvIndex(raxml_ops[i].child2_clv_index, raxml_utree);

		ASSERT_EQ(isLeafNode(parent_network), isLeafNode(parent_raxml));
		ASSERT_EQ(isLeafNode(child1_network), isLeafNode(child1_raxml));
		ASSERT_EQ(isLeafNode(child2_network), isLeafNode(child2_raxml));

		ASSERT_EQ(network_ops[i].parent_clv_index, raxml_ops[i].parent_clv_index);
		ASSERT_EQ(network_ops[i].child1_clv_index, raxml_ops[i].child1_clv_index);
		ASSERT_EQ(network_ops[i].child2_clv_index, raxml_ops[i].child2_clv_index);
		ASSERT_EQ(network_ops[i].parent_scaler_index, raxml_ops[i].parent_scaler_index);
		ASSERT_EQ(network_ops[i].child1_scaler_index, raxml_ops[i].child1_scaler_index);
		ASSERT_EQ(network_ops[i].child2_scaler_index, raxml_ops[i].child2_scaler_index);
		ASSERT_EQ(network_ops[i].child1_matrix_index, raxml_ops[i].child1_matrix_index);
		ASSERT_EQ(network_ops[i].child2_matrix_index, raxml_ops[i].child2_matrix_index);
	}
}

TEST_F (LikelihoodTest, simpleNetworkNoRepeatsOnlyDisplayedTreeWithRaxml) {
	Network network = readNetworkFromFile(networkPath);

	/* get partitions assigned to the current thread */
	PartitionAssignment &part_assign = smallInstance.proc_part_assign.at(ParallelContext::proc_id());

	Tree tree(*(displayed_tree_to_utree(network, 0)));

	TreeInfo raxml_treeinfo = TreeInfo(smallInstance.opts, tree, *(smallInstance.parted_msa.get()),
			smallInstance.tip_msa_idmap, part_assign);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 3 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleNetworkWithRepeatsOnlyDisplayedTreeWithRaxml) {
	Network network = readNetworkFromFile(networkPath);

	RaxmlInstance instance = createStandardRaxmlInstance(networkPath, msaPath, true);
	/* get partitions assigned to the current thread */
	PartitionAssignment &part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	Tree tree(*(displayed_tree_to_utree(network, 0)));

	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, tree, *(instance.parted_msa.get()), instance.tip_msa_idmap,
			part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 4 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleTreeNoRepeats) {
	TreeInfo network_treeinfo_tree = createFakeRaxmlTreeinfo(treeInstance, treeNetwork);
	double network_logl = network_treeinfo_tree.loglh(false);
	std::cout << "The computed network_logl 2 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleTreeWithRepeats) {
	Network network = readNetworkFromFile(treePath);

	RaxmlInstance instance = createStandardRaxmlInstance(treePath, msaPath);
	TreeInfo raxml_treeinfo = createFakeRaxmlTreeinfo(instance, network, true);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 5 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleNetworkNoRepeats) {
	Network network = readNetworkFromFile(networkPath);
	RaxmlInstance instance = createStandardRaxmlInstance(networkPath, msaPath);
	TreeInfo raxml_treeinfo = createFakeRaxmlTreeinfo(instance, network, false);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 6 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleNetworkWithRepeats) {
	Network network = readNetworkFromFile(networkPath);

	RaxmlInstance instance = createStandardRaxmlInstance(networkPath, msaPath);
	TreeInfo raxml_treeinfo = createFakeRaxmlTreeinfo(instance, network, true);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 7 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, DISABLED_celineNetwork) {
	std::string input =
			"((protopterus:0.0,(Xenopus:0.0,(((((Monodelphis:0.0,(python:0.0)#H1:0.0):0.0,(Caretta:0.0)#H2:0.0):0.0,(Homo:0.0)#H3:0.0):0.0,(Ornithorhynchus:0.0)#H4:0.0):0.0,(((#H1:0.0,((#H3:0.0,Anolis:0.0):0.0,(Gallus:0.0)#H5:0.0):0.0):0.0,(Podarcis:0.0)#H6:0.0):0.0,(((#H5:0.0,(#H6:0.0,Taeniopygia:0.0):0.0):0.0,(alligator:0.0,Caiman:0.0):0.0):0.0,(phrynops:0.0,(Emys:0.0,((Chelonoidi:0.0,#H4:0.0):0.0,#H2:0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0);";
	Network network = readNetworkFromString(input);
	Options raxml_opts;
	PartitionedMSA parted_msa;
	IDVector tip_msa_idmap;
	PartitionAssignment part_assign;

	TreeInfo raxml_treeinfo = create_fake_raxml_treeinfo(network, raxml_opts, parted_msa, tip_msa_idmap, part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 8 is: " << network_logl << "\n";
	ASSERT_TRUE(true);
}
