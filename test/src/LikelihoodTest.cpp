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

Options createDefaultOptions() {
	Options opts;
	/* if no command specified, default to --search (or --help if no args were given) */
	opts.command = Command::search;
	opts.start_trees.clear();
	opts.random_seed = (long) time(NULL);

	/* compress alignment patterns by default */
	opts.use_pattern_compression = true;

	/* do not use tip-inner case optimization by default */
	opts.use_tip_inner = false;
	/* use site repeats */
	opts.use_repeats = true;

	/* do not use per-rate-category CLV scalers */
	opts.use_rate_scalers = false;

	/* use probabilistic MSA _if available_ (e.g. CATG file was provided) */
	opts.use_prob_msa = true;

	/* optimize model and branch lengths */
	opts.optimize_model = true;
	opts.optimize_brlen = true;

	/* initialize LH epsilon with default value */
	opts.lh_epsilon = DEF_LH_EPSILON;

	/* default: autodetect best SPR radius */
	opts.spr_radius = -1;
	opts.spr_cutoff = 1.0;

	/* bootstrapping / bootstopping */
	opts.bs_metrics.push_back(BranchSupportMetric::fbp);
	opts.bootstop_criterion = BootstopCriterion::autoMRE;
	opts.bootstop_cutoff = RAXML_BOOTSTOP_CUTOFF;
	opts.bootstop_interval = RAXML_BOOTSTOP_INTERVAL;
	opts.bootstop_permutations = RAXML_BOOTSTOP_PERMUTES;

	/* default: linked branch lengths */
	opts.brlen_linkage = PLLMOD_COMMON_BRLEN_SCALED;
	opts.brlen_min = RAXML_BRLEN_MIN;
	opts.brlen_max = RAXML_BRLEN_MAX;

	/* use all available cores per default */
#if defined(_RAXML_PTHREADS) && !defined(_RAXML_MPI)
	opts.num_threads = std::max(1u, sysutil_get_cpu_cores());
#else
	opts.num_threads = 1;
#endif

#if defined(_RAXML_MPI)
	opts.thread_pinning = ParallelContext::ranks_per_node() == 1 ? true : false;
#else
	opts.thread_pinning = false;
#endif

	opts.model_file = "";
	opts.tree_file = "";

	// autodetect CPU instruction set and use respective SIMD kernels
	opts.simd_arch = sysutil_simd_autodetect();
	opts.load_balance_method = LoadBalancing::benoit;

	opts.num_searches = 0;
	opts.num_bootstraps = 0;

	opts.force_mode = false;
	opts.safety_checks = SafetyCheck::all;

	opts.redo_mode = false;
	opts.nofiles_mode = false;

	opts.tbe_naive = false;

	opts.data_type = DataType::autodetect;
	opts.use_prob_msa = false;
	opts.brlen_opt_method = PLLMOD_OPT_BLO_NEWTON_FAST;

	opts.model_file = "DNA";

	return opts;
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

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = treePath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = false;
	instance.opts.use_tip_inner = true;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());
	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, Tree::loadFromFile(instance.opts.tree_file), *(instance.parted_msa.get()),
			instance.tip_msa_idmap, part_assign);

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
	std::string networkPath = "examples/sample_networks/tree.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = networkPath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = false;
	instance.opts.use_tip_inner = true;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, Tree::loadFromFile(instance.opts.tree_file), *(instance.parted_msa.get()),
			instance.tip_msa_idmap, part_assign);

	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 1 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, compareOperationArrays) {
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";
	unetwork_t * unetwork = unetwork_parse_newick(treePath.c_str());
	Network network = convertNetwork(*unetwork);
	pll_utree_t * network_utree = displayed_tree_to_utree(network, 0);

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = treePath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = false;
	instance.opts.use_tip_inner = true;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());
	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, Tree::loadFromFile(instance.opts.tree_file), *(instance.parted_msa.get()),
			instance.tip_msa_idmap, part_assign);
	TreeInfo network_treeinfo = create_fake_raxml_treeinfo(network, instance.opts, *(instance.parted_msa.get()), instance.tip_msa_idmap,
			part_assign);
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

TEST (LikelihoodTest, simpleTreeNoRepeats) {
	std::string networkPath = "examples/sample_networks/tree.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = networkPath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = false;
	instance.opts.use_tip_inner = true;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	TreeInfo raxml_treeinfo = create_fake_raxml_treeinfo(network, instance.opts, *(instance.parted_msa.get()), instance.tip_msa_idmap,
			part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 2 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, simpleNetworkNoRepeatsOnlyDisplayedTreeWithRaxml) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = networkPath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = false;
	instance.opts.use_tip_inner = true;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	Tree tree(*(displayed_tree_to_utree(network, 0)));

	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, tree, *(instance.parted_msa.get()), instance.tip_msa_idmap, part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 3 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, simpleNetworkWithRepeatsOnlyDisplayedTreeWithRaxml) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = networkPath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = true;
	instance.opts.use_tip_inner = false;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	Tree tree(*(displayed_tree_to_utree(network, 0)));

	TreeInfo raxml_treeinfo = TreeInfo(instance.opts, tree, *(instance.parted_msa.get()), instance.tip_msa_idmap, part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 4 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, simpleTreeWithRepeats) {
	std::string networkPath = "examples/sample_networks/tree.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = networkPath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;

	instance.opts.use_repeats = true;
	instance.opts.use_tip_inner = false;

	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	TreeInfo raxml_treeinfo = create_fake_raxml_treeinfo(network, instance.opts, *(instance.parted_msa.get()), instance.tip_msa_idmap,
			part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 5 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, simpleNetworkNoRepeats) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = networkPath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = false;
	instance.opts.use_tip_inner = true;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	TreeInfo raxml_treeinfo = create_fake_raxml_treeinfo(network, instance.opts, *(instance.parted_msa.get()), instance.tip_msa_idmap,
			part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl 6 is: " << network_logl << "\n";
	ASSERT_NE(network_logl, -std::numeric_limits<double>::infinity());
}

TEST (LikelihoodTest, simpleNetworkWithRepeats) {
	std::string networkPath = "examples/sample_networks/small.nw";
	Network network = readNetworkFromFile(networkPath);
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = networkPath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;

	instance.opts.use_repeats = true;
	instance.opts.use_tip_inner = false;

	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	/* get partitions assigned to the current thread */
	PartitionAssignment& part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());

	TreeInfo raxml_treeinfo = create_fake_raxml_treeinfo(network, instance.opts, *(instance.parted_msa.get()), instance.tip_msa_idmap,
			part_assign);
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
