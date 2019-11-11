/*
 * NetworkIOTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>
#include "src/graph/Common.hpp"

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

using namespace netrax;

std::mutex g_singleThread;

class LikelihoodTest: public ::testing::Test {
protected:
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string networkPath = "examples/sample_networks/small.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	Network treeNetwork;
	Network smallNetwork;
	pll_utree_t *raxml_utree;

	std::unique_ptr<RaxmlWrapper> treeWrapper;
	std::unique_ptr<RaxmlWrapper> treeWrapperRepeats;
	std::unique_ptr<RaxmlWrapper> smallWrapper;
	std::unique_ptr<RaxmlWrapper> smallWrapperRepeats;

	virtual void SetUp() {
		g_singleThread.lock();
		NetraxOptions treeOptions;
		treeOptions.network_file = treePath;
		treeOptions.msa_file = msaPath;
		treeOptions.use_repeats = false;

		NetraxOptions treeOptionsRepeats;
		treeOptionsRepeats.network_file = treePath;
		treeOptionsRepeats.msa_file = msaPath;
		treeOptionsRepeats.use_repeats = true;

		NetraxOptions smallOptions;
		smallOptions.network_file = networkPath;
		smallOptions.msa_file = msaPath;
		smallOptions.use_repeats = false;

		NetraxOptions smallOptionsRepeats;
		smallOptionsRepeats.network_file = networkPath;
		smallOptionsRepeats.msa_file = msaPath;
		smallOptionsRepeats.use_repeats = true;

		treeNetwork = readNetworkFromFile(treePath);
		smallNetwork = readNetworkFromFile(networkPath);
		raxml_utree = Tree::loadFromFile(treePath).pll_utree_copy();

		treeWrapper = std::make_unique<RaxmlWrapper>(treeOptions);
		treeWrapperRepeats = std::make_unique<RaxmlWrapper>(treeOptionsRepeats);
		smallWrapper = std::make_unique<RaxmlWrapper>(smallOptions);
		smallWrapperRepeats = std::make_unique<RaxmlWrapper>(smallOptionsRepeats);
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

int cb_trav_all(pll_unode_t* node) {
	(pll_unode_t*) node;
	return 1;
}

std::unordered_set<std::string> collect_tip_labels_utree(pll_utree_t* utree) {
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

bool no_clv_indices_equal(pll_utree_t* utree) {
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
	pll_utree_t *utree = displayed_tree_to_utree(smallNetwork, 0);
	ASSERT_NE(utree, nullptr);

	pll_utree_t *utree2 = displayed_tree_to_utree(smallNetwork, 0);
	ASSERT_NE(utree2, nullptr);

	// compare tip labels
	std::unordered_set<std::string> tip_labels_utree = collect_tip_labels_utree(utree);
	ASSERT_EQ(tip_labels_utree.size(), smallNetwork.num_tips());
	for (size_t i = 0; i < smallNetwork.tip_nodes.size(); ++i) {
		ASSERT_TRUE(tip_labels_utree.find(smallNetwork.tip_nodes[i]->label) != tip_labels_utree.end());
	}
	// compare tip labels for second tree
	std::unordered_set<std::string> tip_labels_utree2 = collect_tip_labels_utree(utree2);
	ASSERT_EQ(tip_labels_utree2.size(), smallNetwork.num_tips());
	for (size_t i = 0; i < smallNetwork.tip_nodes.size(); ++i) {
		ASSERT_TRUE(tip_labels_utree2.find(smallNetwork.tip_nodes[i]->label) != tip_labels_utree2.end());
	}

	// check for all different clvs
	ASSERT_TRUE(no_clv_indices_equal(utree));

	// check for all different clvs for second tree
	ASSERT_TRUE(no_clv_indices_equal(utree2));

	// TODO: check the branch lengths!!!
}

TEST_F (LikelihoodTest, simpleTreeNoRepeatsNormalRaxml) {
	TreeInfo raxml_treeinfo = treeWrapper->createRaxmlTreeinfo(raxml_utree);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 1 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

void compare_clv(double* clv_raxml, double* clv_network, size_t clv_size) {
	for (size_t i = 0; i < clv_size; ++i) {
		ASSERT_EQ(clv_raxml[i], clv_network[i]);
	}
}

void comparePartitions(const pll_partition_t *p_network, const pll_partition_t *p_raxml) {
	ASSERT_EQ(p_network->tips, p_raxml->tips);
	ASSERT_EQ(p_network->clv_buffers, p_raxml->clv_buffers + 1);
	ASSERT_EQ(p_network->nodes, p_raxml->nodes + 1);
	ASSERT_EQ(p_network->states, p_raxml->states);
	ASSERT_EQ(p_network->sites, p_raxml->sites);
	ASSERT_EQ(p_network->pattern_weight_sum, p_raxml->pattern_weight_sum);
	ASSERT_EQ(p_network->rate_matrices, p_raxml->rate_matrices);
	ASSERT_EQ(p_network->prob_matrices, p_raxml->prob_matrices + 1);
	ASSERT_EQ(p_network->rate_cats, p_raxml->rate_cats);
	ASSERT_EQ(p_network->scale_buffers, p_raxml->scale_buffers + 1);
	ASSERT_EQ(p_network->attributes, p_raxml->attributes);
	ASSERT_EQ(p_network->alignment, p_raxml->alignment);
	ASSERT_EQ(p_network->states_padded, p_raxml->states_padded);

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
	TreeInfo network_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(treeNetwork);
	TreeInfo raxml_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(raxml_utree);

	const pllmod_treeinfo_t &network_treeinfo = network_treeinfo_tree.pll_treeinfo();
	const pllmod_treeinfo_t &raxml_treeinfo = raxml_treeinfo_tree.pll_treeinfo();

	ASSERT_EQ(network_treeinfo.active_partition, raxml_treeinfo.active_partition);
	ASSERT_EQ(network_treeinfo.brlen_linkage, raxml_treeinfo.brlen_linkage);
	ASSERT_EQ(network_treeinfo.init_partition_count, raxml_treeinfo.init_partition_count);
	ASSERT_EQ(network_treeinfo.partition_count, raxml_treeinfo.partition_count);
	ASSERT_EQ(network_treeinfo.subnode_count, raxml_treeinfo.subnode_count + 3);
	ASSERT_EQ(network_treeinfo.tip_count, raxml_treeinfo.tip_count);

	for (size_t i = 0; i < raxml_treeinfo.partition_count; ++i) {
		comparePartitions(network_treeinfo.partitions[i], raxml_treeinfo.partitions[i]);
	}

	double network_logl = network_treeinfo_tree.loglh();
	double raxml_logl = raxml_treeinfo_tree.loglh();

	for (size_t i = 0; i < raxml_treeinfo.partition_count; ++i) {
		comparePartitions(network_treeinfo.partitions[i], raxml_treeinfo.partitions[i]);
	}

	ASSERT_EQ(network_logl, raxml_logl);
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

	TreeInfo raxml_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(raxml_utree);

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
	TreeInfo raxml_treeinfo = treeWrapper->createRaxmlTreeinfo(displayed_tree_to_utree(smallNetwork, 0));

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 3 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleNetworkWithRepeatsOnlyDisplayedTreeWithRaxml) {
	pll_utree_t *network_utree = displayed_tree_to_utree(smallNetwork, 0);

	TreeInfo raxml_treeinfo = treeWrapperRepeats->createRaxmlTreeinfo(displayed_tree_to_utree(smallNetwork, 0));

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 4 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleTreeNoRepeats) {
	TreeInfo network_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(treeNetwork);
	double network_logl = network_treeinfo_tree.loglh(false);
	std::cout << "The computed network_logl 2 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, likelihoodFunctionsTree) {
	pll_utree_t* utree = displayed_tree_to_utree(treeNetwork, 0);
	TreeInfo raxml_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(utree);
	double raxml_logl = raxml_treeinfo_tree.loglh(0);
	std::cout << "raxml logl: " << raxml_logl << "\n";

	TreeInfo network_treeinfo_tree = treeWrapper->createRaxmlTreeinfo(treeNetwork);
	double naive_logl = computeLoglikelihoodNaiveUtree(*(treeWrapper.get()), treeNetwork, 0, 1);
	std::cout << "naive logl: " << naive_logl << "\n";

	RaxmlWrapper::NetworkParams* params = (RaxmlWrapper::NetworkParams*) network_treeinfo_tree.pll_treeinfo().likelihood_computation_params;

	double sarah_logl = computeLoglikelihood(treeNetwork, *(params->network_treeinfo), 0, 1);
	std::cout << "sarah logl: " << sarah_logl << "\n";

	double norep_logl = computeLoglikelihoodLessExponentiation(treeNetwork, *(params->network_treeinfo), 0, 1);
	std::cout << "norep_logl: " << norep_logl << "\n";

	ASSERT_EQ(raxml_logl, naive_logl);
	ASSERT_EQ(naive_logl, sarah_logl);
	ASSERT_EQ(naive_logl, norep_logl);
}

TEST_F (LikelihoodTest, likelihoodFunctionsNetwork) {
	TreeInfo network_treeinfo_small = smallWrapper->createRaxmlTreeinfo(smallNetwork);
	RaxmlWrapper::NetworkParams* params = (RaxmlWrapper::NetworkParams*) network_treeinfo_small.pll_treeinfo().likelihood_computation_params;

	double naive_logl = computeLoglikelihoodNaiveUtree(*(smallWrapper.get()), smallNetwork, 0, 1);
	std::cout << "naive logl: " << naive_logl << "\n";

	double sarah_logl = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1);
	std::cout << "sarah logl: " << sarah_logl << "\n";

	double norep_logl = computeLoglikelihoodLessExponentiation(smallNetwork, *(params->network_treeinfo), 0, 1);
	std::cout << "norep_logl: " << norep_logl << "\n";

	ASSERT_EQ(naive_logl, sarah_logl);
	ASSERT_NE(norep_logl, -std::numeric_limits<double>::infinity());
	//ASSERT_EQ(sarah_logl, norep_logl);
}

TEST_F (LikelihoodTest, updateReticulationProb) {
	TreeInfo network_treeinfo_small = smallWrapper->createRaxmlTreeinfo(smallNetwork);
	RaxmlWrapper::NetworkParams* params = (RaxmlWrapper::NetworkParams*) network_treeinfo_small.pll_treeinfo().likelihood_computation_params;

	double sarah_logl = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, true);
	std::cout << "sarah logl: " << sarah_logl << "\n";
	double sarah_logl_2 = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, true);
	std::cout << "sarah logl_2: " << sarah_logl_2 << "\n";
	double sarah_logl_3 = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, false);
	std::cout << "sarah logl_3: " << sarah_logl_3 << "\n";

	double norep_logl = computeLoglikelihoodLessExponentiation(smallNetwork, *(params->network_treeinfo), 0, 1, true);
	std::cout << "norep_logl: " << norep_logl << "\n";
	double norep_logl_2 = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, true);
	std::cout << "norep logl_2: " << norep_logl_2 << "\n";
	double norep_logl_3 = computeLoglikelihood(smallNetwork, *(params->network_treeinfo), 0, 1, false);
	std::cout << "norep logl_3: " << norep_logl_3 << "\n";

	ASSERT_EQ(sarah_logl_2, sarah_logl_3);
	ASSERT_EQ(norep_logl_2, norep_logl_3);

	/*ASSERT_EQ(sarah_logl, norep_logl);
	ASSERT_EQ(sarah_logl_2, norep_logl_2);
	ASSERT_EQ(sarah_logl_3, norep_logl_3);*/
}

TEST_F (LikelihoodTest, simpleTreeWithRepeats) {
	TreeInfo network_treeinfo_tree = treeWrapperRepeats->createRaxmlTreeinfo(treeNetwork);
	double network_logl = network_treeinfo_tree.loglh(false);
	std::cout << "The computed network_logl 5 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleNetworkNoRepeats) {
	TreeInfo network_treeinfo = smallWrapper->createRaxmlTreeinfo(smallNetwork);
	double network_logl = network_treeinfo.loglh(false);
	std::cout << "The computed network_logl 6 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, simpleNetworkWithRepeats) {
	TreeInfo network_treeinfo = smallWrapperRepeats->createRaxmlTreeinfo(smallNetwork);
	double network_logl = network_treeinfo.loglh(false);
	std::cout << "The computed network_logl 7 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, celineNetwork) {
	std::string networkPath = "examples/sample_networks/celine.nw";
	std::string msaPath = "examples/sample_networks/celine_fake_alignment.txt";
	Network network = netrax::readNetworkFromFile(networkPath);
	NetraxOptions options;
	options.network_file = networkPath;
	options.msa_file = msaPath;
	RaxmlWrapper wrapper(options);
	TreeInfo treeInfo = wrapper.createRaxmlTreeinfo(network);

	RaxmlWrapper::NetworkParams* params = (RaxmlWrapper::NetworkParams*) treeInfo.pll_treeinfo().likelihood_computation_params;

	double naive_logl = computeLoglikelihoodNaiveUtree(wrapper, network, 0, 1);
	std::cout << "naive logl: " << naive_logl << "\n";

	double sarah_logl = computeLoglikelihood(network, *(params->network_treeinfo), 0, 1);
	std::cout << "sarah logl: " << sarah_logl << "\n";

	double norep_logl = computeLoglikelihoodLessExponentiation(network, *(params->network_treeinfo), 0, 1);
	std::cout << "norep_logl: " << norep_logl << "\n";

	ASSERT_EQ(naive_logl, sarah_logl);
	ASSERT_NE(norep_logl, -std::numeric_limits<double>::infinity());
}

TEST_F (LikelihoodTest, celineNetworkNonzeroBranches) {
	std::string networkPath = "examples/sample_networks/celine_nonzero_branches.nw";
	std::string msaPath = "examples/sample_networks/celine_fake_alignment.txt";
	Network network = netrax::readNetworkFromFile(networkPath);
	NetraxOptions options;
	options.network_file = networkPath;
	options.msa_file = msaPath;
	RaxmlWrapper wrapper(options);
	TreeInfo treeInfo = wrapper.createRaxmlTreeinfo(network);

	RaxmlWrapper::NetworkParams* params = (RaxmlWrapper::NetworkParams*) treeInfo.pll_treeinfo().likelihood_computation_params;

	double norep_logl = computeLoglikelihoodLessExponentiation(network, *(params->network_treeinfo), 0, 1);
	std::cout << "norep_logl: " << norep_logl << "\n";

	ASSERT_NE(norep_logl, -std::numeric_limits<double>::infinity());
}
