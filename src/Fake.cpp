/*
 * Fake.cpp
 *
 *  Created on: Sep 9, 2019
 *      Author: Sarah Lutteropp
 */

#include "Fake.hpp"

#include <stddef.h>
#include <stdlib.h>
#include <cstring>
#include <stdexcept>
#include <string>

#include "likelihood/LikelihoodComputation.hpp"
#include "optimization/BranchLengthOptimization.hpp"

namespace netrax {

/*
 * Be aware of this one:
 *
 *In treeinfo_init_partition:
 *  // compute some derived dimensions
 unsigned int inner_nodes_count = treeinfo->tip_count - 2;
 unsigned int nodes_count       = inner_nodes_count + treeinfo->tip_count;
 unsigned int branch_count      = nodes_count - 1;
 unsigned int pmatrix_count     = branch_count;
 unsigned int utree_count       = inner_nodes_count * 3 + treeinfo->tip_count;

 in treeinfo_create:
 //compute some derived dimensions
 unsigned int inner_nodes_count = treeinfo->tree->inner_count;
 unsigned int nodes_count       = inner_nodes_count + tips;
 unsigned int branch_count      = treeinfo->tree->edge_count;
 treeinfo->subnode_count        = tips + 3 * inner_nodes_count;


 unsigned int clv_count = treeinfo->tip_count + (treeinfo->tip_count - 2) * 3;
 unsigned int pmatrix_count = treeinfo->tree->edge_count;


 allnodes_count = (treeinfo->tip_count - 2) * 3;


 It should be like this one:
 in networkinfo_create:
 // compute some derived dimensions
 unsigned int inner_nodes_count = networkinfo->network->inner_tree_count + networkinfo->network->reticulation_count;
 unsigned int nodes_count = inner_nodes_count + tips;
 unsigned int branch_count = networkinfo->network->edge_count;
 networkinfo->subnode_count = tips + 3 * inner_nodes_count;

 in networkinfo_init_partition:
 // compute some derived dimensions
 unsigned int inner_nodes_count = networkinfo->network->tip_count - 2 + MAX_RETICULATION_COUNT + 1; // +1 for the fake extra entry
 unsigned int nodes_count = inner_nodes_count + networkinfo->network->tip_count;
 unsigned int branch_count = nodes_count - 1;
 unsigned int pmatrix_count = branch_count;
 unsigned int unetwork_count = inner_nodes_count * 3 + networkinfo->network->tip_count;


 */

void fake_init_collect_branch_lengths(pllmod_treeinfo_t *treeinfo,
		Network &network) {
	// collect the branch lengths
	for (size_t i = 0; i < network.edges.size(); ++i) {
		treeinfo->branch_lengths[0][i] = network.edges[i].getLength();
	}

	treeinfo->branch_lengths[0][network.edges.size()] = 0; // the fake branch length

	/* in unlinked branch length mode, copy brlen to other partitions */
	if (treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
		for (unsigned int i = 1; i < treeinfo->partition_count; ++i) {
			// TODO: only save brlens for initialized partitions
			//      if (treeinfo->partitions[i])
			{
				memcpy(treeinfo->branch_lengths[i], treeinfo->branch_lengths[0],
						treeinfo->tree->edge_count * sizeof(double));
			}
		}
	}
}

int fake_init_tree(pllmod_treeinfo_t *treeinfo, Network &network) {
	pll_utree_t *tree = (pll_utree_t*) malloc(sizeof(pll_utree_t));
	treeinfo->tree = tree;

	tree->tip_count = network.num_tips();
	tree->edge_count = network.num_branches() + 1; // +1 for the fake pmatrix index
	tree->inner_count = network.num_inner() + 1; // +1 for the fake clv index

	tree->nodes = NULL;
	tree->vroot = NULL;

	treeinfo->root = NULL;

	return PLL_SUCCESS;
}

void destroy_fake_treeinfo(pllmod_treeinfo_t *treeinfo) {
	if (!treeinfo)
		return;
	if (treeinfo->likelihood_computation_params != treeinfo) {
		free(treeinfo->likelihood_computation_params);
	}
	pllmod_treeinfo_destroy(treeinfo);
}

double fake_network_loglikelihood(void *network_params, int incremental,
		int update_pmatrices) {
	NetworkParams *params = (NetworkParams*) network_params;
	return computeLoglikelihood(*params->network, *params->fake_treeinfo,
			incremental, update_pmatrices);
}

double fake_opt_brlen(pllmod_treeinfo_t *fake_treeinfo, double min_brlen,
		double max_brlen, double lh_epsilon, int max_iters, int opt_method,
		int radius) {
	Network *network =
			((NetworkParams*) (fake_treeinfo->likelihood_computation_params))->network;
	return optimize_branches(*network, *fake_treeinfo, min_brlen, max_brlen,
			lh_epsilon, max_iters, opt_method, radius);
}

double fake_spr_round(pllmod_treeinfo_t *treeinfo, unsigned int radius_min,
		unsigned int radius_max, unsigned int ntopol_keep, pll_bool_t thorough,
		int brlen_opt_method, double bl_min, double bl_max, int smoothings,
		double epsilon, cutoff_info_t *cutoff_info, double subtree_cutoff) {
	throw std::runtime_error("Not implemented yet");
}

pllmod_ancestral_t* fake_compute_ancestral(pllmod_treeinfo_t *treeinfo) {
	throw std::runtime_error("Not implemented yet");
}

pllmod_treeinfo_t* create_fake_treeinfo(Network &network, unsigned int tips,
		unsigned int partitions, int brlen_linkage) {
	assert(partitions > 0);
	/* create treeinfo instance */
	pllmod_treeinfo_t *treeinfo;

	if (!(treeinfo = (pllmod_treeinfo_t*) calloc(1, sizeof(pllmod_treeinfo_t)))) {
		throw std::runtime_error("Cannot allocate memory for treeinfo\n");
		return NULL;
	}

	/* save dimensions & options */
	treeinfo->tip_count = tips;
	treeinfo->partition_count = partitions;
	treeinfo->brlen_linkage = brlen_linkage;

	fake_init_tree(treeinfo, network);

	/* compute some derived dimensions */
	unsigned int inner_nodes_count = treeinfo->tree->inner_count;
	unsigned int nodes_count = inner_nodes_count + tips;
	unsigned int branch_count = treeinfo->tree->edge_count;
	treeinfo->subnode_count = tips + 3 * inner_nodes_count;

	treeinfo->travbuffer = NULL;
	treeinfo->matrix_indices = NULL;
	treeinfo->operations = NULL;
	treeinfo->subnodes = NULL;

	/* allocate arrays for storing per-partition info */
	treeinfo->partitions = (pll_partition_t**) calloc(partitions,
			sizeof(pll_partition_t*));
	treeinfo->params_to_optimize = (int*) calloc(partitions, sizeof(int));
	treeinfo->alphas = (double*) calloc(partitions, sizeof(double));
	treeinfo->gamma_mode = (int*) calloc(partitions, sizeof(int));
	treeinfo->param_indices = (unsigned int**) calloc(partitions,
			sizeof(unsigned int*));
	treeinfo->subst_matrix_symmetries = (int**) calloc(partitions,
			sizeof(int*));
	treeinfo->branch_lengths = (double**) calloc(partitions, sizeof(double*));
	treeinfo->deriv_precomp = (double**) calloc(partitions, sizeof(double*));
	treeinfo->clv_valid = (char**) calloc(partitions, sizeof(char*));
	treeinfo->pmatrix_valid = (char**) calloc(partitions, sizeof(char*));
	treeinfo->partition_loglh = (double*) calloc(partitions, sizeof(double));

	treeinfo->init_partition_count = 0;
	treeinfo->init_partition_idx = (unsigned int*) calloc(partitions,
			sizeof(unsigned int));
	treeinfo->init_partitions = (pll_partition_t**) calloc(partitions,
			sizeof(pll_partition_t*));

	/* allocate array for storing linked/average branch lengths */
	treeinfo->linked_branch_lengths = (double*) malloc(
			branch_count * sizeof(double));

	/* allocate branch length scalers if needed */
	if (brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED)
		treeinfo->brlen_scalers = (double*) calloc(partitions, sizeof(double));
	else
		treeinfo->brlen_scalers = NULL;

	/* check memory allocation */
	if (!treeinfo->partitions || !treeinfo->alphas || !treeinfo->param_indices
			|| !treeinfo->subst_matrix_symmetries || !treeinfo->branch_lengths
			|| !treeinfo->deriv_precomp || !treeinfo->clv_valid
			|| !treeinfo->pmatrix_valid || !treeinfo->linked_branch_lengths
			|| !treeinfo->partition_loglh || !treeinfo->gamma_mode
			|| !treeinfo->init_partition_idx
			|| (brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED
					&& !treeinfo->brlen_scalers)) {
		throw std::runtime_error(
				"Cannot allocate memory for treeinfo arrays\n");
		return NULL;
	}

	unsigned int p;
	for (p = 0; p < partitions; ++p) {
		/* use mean GAMMA rates per default */
		treeinfo->gamma_mode[p] = PLL_GAMMA_RATES_MEAN;

		/* allocate arrays for storing the per-partition branch lengths */
		if (brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
			treeinfo->branch_lengths[p] = (double*) malloc(
					branch_count * sizeof(double));
		} else
			treeinfo->branch_lengths[p] = treeinfo->linked_branch_lengths;

		/* initialize all branch length scalers to 1 */
		if (treeinfo->brlen_scalers)
			treeinfo->brlen_scalers[p] = 1.;

		/* check memory allocation */
		if (!treeinfo->branch_lengths[p]) {
			throw std::runtime_error(
					"Cannot allocate memory for arrays for partition "
							+ std::to_string(p));
			return NULL;
		}
	}

	/* by default, work with all partitions */
	treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;

	NetworkParams *params = (NetworkParams*) malloc(sizeof(NetworkParams));
	params->network = &network;
	params->fake_treeinfo = treeinfo;
	treeinfo->likelihood_target_function = fake_network_loglikelihood;
	treeinfo->likelihood_computation_params = (void*) params;

	fake_init_collect_branch_lengths(treeinfo, network);

	return treeinfo;
}

TreeInfo create_fake_raxml_treeinfo(Network &network, const Options &opts,
		const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
		const PartitionAssignment &part_assign,
		const std::vector<uintVector> &site_weights) {
	pllmod_treeinfo_t *base_treeinfo;

	base_treeinfo = create_fake_treeinfo(network, network.num_tips(), parted_msa.part_count(), opts.brlen_linkage);

	/*std::cout << "just for debugging purposes, creating a normal treeinfo.\n";
	pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, 0);

	base_treeinfo = pllmod_treeinfo_create(displayed_tree->vroot,
			displayed_tree->tip_count, parted_msa.part_count(),
			opts.brlen_linkage);*/

	return TreeInfo(opts, base_treeinfo, parted_msa, tip_msa_idmap, part_assign,
			site_weights, fake_opt_brlen, fake_spr_round,
			fake_compute_ancestral, destroy_fake_treeinfo, network.num_tips(),
			network.num_inner(), network.num_branches());
}

TreeInfo create_fake_raxml_treeinfo(Network &network, const Options &opts,
		const std::vector<doubleVector> &partition_brlens,
		const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
		const PartitionAssignment &part_assign) {
	return create_fake_raxml_treeinfo(network, opts, parted_msa, tip_msa_idmap,
			part_assign, std::vector<uintVector>());
}
TreeInfo create_fake_raxml_treeinfo(Network &network, const Options &opts,
		const std::vector<doubleVector> &partition_brlens,
		const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
		const PartitionAssignment &part_assign,
		const std::vector<uintVector> &site_weights) {
	return create_fake_raxml_treeinfo(network, opts, parted_msa, tip_msa_idmap,
			part_assign, site_weights);
}
TreeInfo create_fake_raxml_treeinfo(Network &network, const Options &opts,
		const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
		const PartitionAssignment &part_assign) {
	return create_fake_raxml_treeinfo(network, opts, parted_msa, tip_msa_idmap,
			part_assign, std::vector<uintVector>());
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

RaxmlInstance createStandardRaxmlInstance(const std::string &treePath,
		const std::string &msaPath, bool useRepeats) {
	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = treePath;
	instance.opts.msa_file = msaPath;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = useRepeats;
	instance.opts.use_tip_inner = !useRepeats;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	return instance;
}

void setInstanceRepeats(RaxmlInstance &instance, bool useRepeats) {
	instance.opts.use_repeats = useRepeats;
	instance.opts.use_tip_inner = !useRepeats;
}

TreeInfo createStandardRaxmlTreeinfo(RaxmlInstance &instance, bool useRepeats) {
	setInstanceRepeats(instance, useRepeats);
	/* get partitions assigned to the current thread */
	PartitionAssignment &part_assign = instance.proc_part_assign.at(
			ParallelContext::proc_id());
	TreeInfo raxml_treeinfo = TreeInfo(instance.opts,
			Tree::loadFromFile(instance.opts.tree_file),
			*(instance.parted_msa.get()), instance.tip_msa_idmap, part_assign);
	return raxml_treeinfo;
}

TreeInfo createFakeRaxmlTreeinfo(RaxmlInstance &instance, Network &network,
		bool useRepeats) {
	setInstanceRepeats(instance, useRepeats);
	/* get partitions assigned to the current thread */
	PartitionAssignment &part_assign = instance.proc_part_assign.at(
			ParallelContext::proc_id());
	TreeInfo network_treeinfo = create_fake_raxml_treeinfo(network,
			instance.opts, *(instance.parted_msa.get()), instance.tip_msa_idmap,
			part_assign);
	return network_treeinfo;
}

}
