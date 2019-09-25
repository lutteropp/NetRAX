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
#include <iostream>
#include "src/Network.hpp"

#include <raxml-ng/main.hpp>

using namespace netrax;

TEST (LikelihoodTest, testTheTest) {
	ASSERT_TRUE(true);
}

std::vector<size_t> getNeighborClvIndices(pll_unode_t* node) {
	std::vector<size_t> neighbors;
	if (node->next) {
		pll_unode_t* actNode = node;
		do {
			neighbors.push_back(actNode->back->clv_index);
			actNode = actNode->next;
		} while (actNode != node);
	}
	return neighbors;
}

void compareNodes(pll_unode_t* node1, pll_unode_t* node2) {
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

TEST (LikelihoodTest, displayedTreeOfTreeToUtree) {
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";
	unetwork_t * unetwork = unetwork_parse_newick(treePath.c_str());
	Network network = convertNetwork(*unetwork);
	pll_utree_t * network_utree = displayed_tree_to_utree(network, 0);

	TreeInfo raxml_treeinfo = createStandardRaxmlTreeinfo(treePath, msaPath, false);

	ASSERT_NE(network_utree, nullptr);
	// compare the utrees:
	pll_utree_t* raxml_utree = raxml_treeinfo.pll_treeinfo().tree;
	ASSERT_EQ(network_utree->inner_count, raxml_utree->inner_count);
	ASSERT_EQ(network_utree->binary, raxml_utree->binary);
	ASSERT_EQ(network_utree->edge_count, raxml_utree->edge_count);
	ASSERT_EQ(network_utree->tip_count, raxml_utree->tip_count);
	compareNodes(network_utree->vroot, raxml_utree->vroot);

	for (size_t i = 0; i < network.nodes.size(); ++i) {
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

TEST (LikelihoodTest, displayedTreeOfNetworkToUtree) {
	std::string treePath = "examples/sample_networks/small.nw";
	unetwork_t * unetwork = unetwork_parse_newick(treePath.c_str());
	Network network = convertNetwork(*unetwork);
	pll_utree_t * utree = displayed_tree_to_utree(network, 0);
	ASSERT_NE(utree, nullptr);
}

TEST (LikelihoodTest, simpleTreeNoRepeatsNormalRaxml) {
	std::string treePath = "examples/sample_networks/tree.nw";
	Network network = readNetworkFromFile(treePath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	TreeInfo raxml_treeinfo = createStandardRaxmlTreeinfo(treePath, msaPath, false);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 1 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, DISABLED_compareOperationArrays) {
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";
	unetwork_t * unetwork = unetwork_parse_newick(treePath.c_str());
	Network network = convertNetwork(*unetwork);
	pll_utree_t * network_utree = displayed_tree_to_utree(network, 0);

	TreeInfo raxml_treeinfo = createStandardRaxmlTreeinfo(treePath, msaPath);
	TreeInfo network_treeinfo = createFakeRaxmlTreeinfo(network, treePath, msaPath);

	raxml_treeinfo.loglh(false); // to fill the operations array

	std::vector<pll_operation_t> network_ops = createOperations(network, 0);
	pll_operation_t* raxml_ops = raxml_treeinfo.pll_treeinfo().operations;

	std::cout << "Number of operations: " << network_ops.size() << "\n";
	for (size_t i = 0; i < network_ops.size(); ++i) {
		std::cout << "network_ops[" << i << "]: \n";
		std::cout << "  parent_clv_index: " << network_ops[i].parent_clv_index << "\n";
		std::cout << "  parent_scaler_index: " << network_ops[i].parent_scaler_index << "\n";
		std::cout << "  child1_clv_index: " << network_ops[i].child1_clv_index << "\n";
		std::cout << "  child1_matrix_index: " << network_ops[i].child1_matrix_index << "\n";
		std::cout << "  child1_scaler_index: " << network_ops[i].child1_scaler_index << "\n";
		std::cout << "  child2_clv_index: " << network_ops[i].child2_clv_index << "\n";
		std::cout << "  child2_matrix_index: " << network_ops[i].child2_matrix_index << "\n";
		std::cout << "  child2_scaler_index: " << network_ops[i].child2_scaler_index << "\n";

		std::cout << "raxml_ops[" << i << "]: \n";
		std::cout << "  parent_clv_index: " << raxml_ops[i].parent_clv_index << "\n";
		std::cout << "  parent_scaler_index: " << raxml_ops[i].parent_scaler_index << "\n";
		std::cout << "  child1_clv_index: " << raxml_ops[i].child1_clv_index << "\n";
		std::cout << "  child1_matrix_index: " << raxml_ops[i].child1_matrix_index << "\n";
		std::cout << "  child1_scaler_index: " << raxml_ops[i].child1_scaler_index << "\n";
		std::cout << "  child2_clv_index: " << raxml_ops[i].child2_clv_index << "\n";
		std::cout << "  child2_matrix_index: " << raxml_ops[i].child2_matrix_index << "\n";
		std::cout << "  child2_scaler_index: " << raxml_ops[i].child2_scaler_index << "\n";
		std::cout << "\n";

		/*ASSERT_EQ(network_ops[i].parent_clv_index, raxml_ops[i].parent_clv_index);
		 ASSERT_EQ(network_ops[i].child1_clv_index, raxml_ops[i].child1_clv_index);
		 ASSERT_EQ(network_ops[i].child2_clv_index, raxml_ops[i].child2_clv_index);
		 ASSERT_EQ(network_ops[i].parent_scaler_index, raxml_ops[i].parent_scaler_index);
		 ASSERT_EQ(network_ops[i].child1_scaler_index, raxml_ops[i].child1_scaler_index);
		 ASSERT_EQ(network_ops[i].child2_scaler_index, raxml_ops[i].child2_scaler_index);
		 ASSERT_EQ(network_ops[i].child1_matrix_index, raxml_ops[i].child1_matrix_index);
		 ASSERT_EQ(network_ops[i].child2_matrix_index, raxml_ops[i].child2_matrix_index);*/
	}

}

TEST (LikelihoodTest, DISABLED_simpleTreeNoRepeats) {
	std::string treePath = "examples/sample_networks/tree.nw";
	Network network = readNetworkFromFile(treePath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	TreeInfo network_treeinfo = createFakeRaxmlTreeinfo(network, treePath, msaPath, false);

	double network_logl = network_treeinfo.loglh(false);
	std::cout << "The computed network_logl 2 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, DISABLED_simpleNetworkNoRepeatsOnlyDisplayedTreeWithRaxml) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance = createStandardRaxmlInstance(networkPath, msaPath, false);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	Tree tree(*(displayed_tree_to_utree(network, 0)));

	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, tree, *(instance.parted_msa.get()), instance.tip_msa_idmap, part_assign);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 3 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, DISABLED_simpleNetworkWithRepeatsOnlyDisplayedTreeWithRaxml) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance = createStandardRaxmlInstance(networkPath, msaPath, true);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	Tree tree(*(displayed_tree_to_utree(network, 0)));

	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, tree, *(instance.parted_msa.get()), instance.tip_msa_idmap, part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 4 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, DISABLED_simpleTreeWithRepeats) {
	std::string treePath = "examples/sample_networks/tree.nw";
	Network network = readNetworkFromFile(treePath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	TreeInfo raxml_treeinfo = createFakeRaxmlTreeinfo(network, treePath, msaPath, true);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 5 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, DISABLED_simpleNetworkNoRepeats) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	TreeInfo raxml_treeinfo = createFakeRaxmlTreeinfo(network, networkPath, msaPath, false);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 6 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, DISABLED_simpleNetworkWithRepeats) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	TreeInfo raxml_treeinfo = createFakeRaxmlTreeinfo(network, networkPath, msaPath, true);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 7 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, DISABLED_celineNetwork) {
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
