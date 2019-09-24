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

TEST (LikelihoodTest, DISABLED_simpleNetwork) {
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
	std::cout << "The computed network_logl is: " << network_logl << "\n";
	ASSERT_TRUE(true);
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
	std::cout << "The computed network_logl is: " << network_logl << "\n";
	ASSERT_TRUE(true);
}
