/*
 * RaxmlWrapper.cpp
 *
 *  Created on: Sep 28, 2019
 *      Author: sarah
 */

#include "RaxmlWrapper.hpp"

#include "likelihood/LikelihoodComputation.hpp"
#include "optimization/BranchLengthOptimization.hpp"

namespace netrax {

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

RaxmlInstance createRaxmlInstance(const NetraxOptions &options) {
	RaxmlInstance instance;
	instance.opts = createDefaultOptions();
	instance.opts.tree_file = options.network_file;
	instance.opts.msa_file = options.msa_file;
	instance.opts.command = Command::evaluate;
	instance.opts.num_threads = 1;
	instance.opts.use_repeats = options.use_repeats;
	instance.opts.use_tip_inner = !options.use_repeats;
	load_parted_msa(instance);
	check_options(instance);
	balance_load(instance);
	return instance;
}

TreeInfo createRaxmlTreeinfo(RaxmlInstance &instance, Network &network) {
	pllmod_treeinfo_t *pllTreeinfo = createNetworkPllTreeinfo(network, network.num_tips(),
			instance.parted_msa->part_count(), instance.opts.brlen_linkage);
	TreeInfo::tinfo_behaviour network_behaviour;
	network_behaviour.compute_ancestral_function = network_ancestral_wrapper;
	network_behaviour.opt_brlen_function = network_opt_brlen_wrapper;
	network_behaviour.spr_round_function = network_spr_round_wrapper;
	network_behaviour.destroy_treeinfo_function = destroy_network_treeinfo;
	network_behaviour.init_function = network_init_treeinfo_wrapper;
	return createRaxmlTreeinfo(instance, pllTreeinfo, network_behaviour);
}

TreeInfo createRaxmlTreeinfo(RaxmlInstance &instance, const pll_utree_t *utree) {
	// Check that the MSA has already been loaded
	assert(!instance.tip_id_map.empty());
	pllmod_treeinfo_t *pllTreeinfo = createStandardPllTreeinfo(utree, instance.parted_msa->part_count(),
			instance.opts.brlen_linkage);
	TreeInfo::tinfo_behaviour standard_behaviour;
	return createRaxmlTreeinfo(instance, pllTreeinfo, standard_behaviour);
}

void set_partition_fake_entry(pll_partition_t* partition, size_t fake_clv_index, size_t fake_pmatrix_index) {
	// set pmatrix to identity for the fake node
	unsigned int states = partition->states;
	unsigned int states_padded = partition->states_padded;
	unsigned int sites = partition->sites;
	unsigned int rate_cats = partition->rate_cats;
	double * pmat = partition->pmatrix[fake_pmatrix_index];
	unsigned int i, j, k;
	for (i = 0; i < rate_cats; ++i) {
		for (j = 0; j < states; ++j) {
			for (k = 0; k < states; ++k)
				pmat[j * states_padded + k] = 1;
		}
		pmat += states * states_padded;
	}

	// set clv to all-ones for the fake node
	double* clv = partition->clv[fake_clv_index];

	if (clv == NULL) { // this happens when we have site repeats
		// TODO: Does it work? Or do we need to increase the number of tips somehow when creating the partition?
		partition->clv[fake_clv_index] = (double*) pll_aligned_alloc(sites * rate_cats * states_padded * sizeof(double),
				partition->alignment);
		clv = partition->clv[fake_clv_index];
	}

	unsigned int n;
	for (n = 0; n < sites; ++n) {
		for (i = 0; i < rate_cats; ++i) {
			for (j = 0; j < states; ++j) {
				clv[j] = 1;
			}

			clv += states_padded;
		}
	}
}

void network_init_treeinfo_wrapper(const Options &opts, const std::vector<doubleVector> &partition_brlens,
		size_t num_branches, const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
		const PartitionAssignment &part_assign, const std::vector<uintVector> &site_weights,
		doubleVector *partition_contributions, pllmod_treeinfo_t *pll_treeinfo, IDSet *parts_master) {

	throw std::runtime_error("not implemented yet");
	// Copy&Paste from standard function follows...

	partition_contributions->resize(parted_msa.part_count());
	double total_weight = 0;

	if (ParallelContext::num_procs() > 1) {
		pllmod_treeinfo_set_parallel_context(pll_treeinfo, (void*) nullptr, ParallelContext::parallel_reduce_cb);
	}

	// init partitions
	int optimize_branches = opts.optimize_brlen ? PLLMOD_OPT_PARAM_BRANCHES_ITERATIVE : 0;

	for (size_t p = 0; p < parted_msa.part_count(); ++p) {
		const PartitionInfo &pinfo = parted_msa.part_info(p);
		const auto &weights = site_weights.empty() ? pinfo.msa().weights() : site_weights.at(p);
		int params_to_optimize = opts.optimize_model ? pinfo.model().params_to_optimize() : 0;
		params_to_optimize |= optimize_branches;

		(*partition_contributions)[p] = std::accumulate(weights.begin(), weights.end(), 0);
		total_weight += (*partition_contributions)[p];

		PartitionAssignment::const_iterator part_range = part_assign.find(p);
		if (part_range != part_assign.end()) {
			/* create and init PLL partition structure */
			pll_partition_t *partition = create_pll_partition(opts, pinfo, tip_msa_idmap, *part_range, weights);

			int retval = pllmod_treeinfo_init_partition(pll_treeinfo, p, partition, params_to_optimize,
					pinfo.model().gamma_mode(), pinfo.model().alpha(), pinfo.model().ratecat_submodels().data(),
					pinfo.model().submodel(0).rate_sym().data());

			if (!retval) {
				assert(pll_errno);
				libpll_check_error("ERROR adding treeinfo partition");
			}

			// set per-partition branch lengths or scalers
			if (opts.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
				assert(pll_treeinfo->brlen_scalers);
				pll_treeinfo->brlen_scalers[p] = pinfo.model().brlen_scaler();
			} else if (opts.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED && !partition_brlens.empty()) {
				assert(pll_treeinfo->branch_lengths[p]);
				memcpy(pll_treeinfo->branch_lengths[p], partition_brlens[p].data(), num_branches * sizeof(double));
			}

			if (part_range->master())
				(*parts_master).insert(p);
		} else {
			// this partition will be processed by other threads, but we still need to know
			// which parameters to optimize
			pll_treeinfo->params_to_optimize[p] = params_to_optimize;
		}
	}

	// finalize partition contribution computation
	for (auto &c : *partition_contributions)
		c /= total_weight;

}

void fake_init_collect_branch_lengths(pllmod_treeinfo_t *treeinfo, const Network &network) {
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

pllmod_treeinfo_t* createNetworkPllTreeinfo(Network &network, unsigned int tips, unsigned int partitions,
		int brlen_linkage) {

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
	treeinfo->partitions = (pll_partition_t**) calloc(partitions, sizeof(pll_partition_t*));
	treeinfo->params_to_optimize = (int*) calloc(partitions, sizeof(int));
	treeinfo->alphas = (double*) calloc(partitions, sizeof(double));
	treeinfo->gamma_mode = (int*) calloc(partitions, sizeof(int));
	treeinfo->param_indices = (unsigned int**) calloc(partitions, sizeof(unsigned int*));
	treeinfo->subst_matrix_symmetries = (int**) calloc(partitions, sizeof(int*));
	treeinfo->branch_lengths = (double**) calloc(partitions, sizeof(double*));
	treeinfo->deriv_precomp = (double**) calloc(partitions, sizeof(double*));
	treeinfo->clv_valid = (char**) calloc(partitions, sizeof(char*));
	treeinfo->pmatrix_valid = (char**) calloc(partitions, sizeof(char*));
	treeinfo->partition_loglh = (double*) calloc(partitions, sizeof(double));

	treeinfo->init_partition_count = 0;
	treeinfo->init_partition_idx = (unsigned int*) calloc(partitions, sizeof(unsigned int));
	treeinfo->init_partitions = (pll_partition_t**) calloc(partitions, sizeof(pll_partition_t*));

	/* allocate array for storing linked/average branch lengths */
	treeinfo->linked_branch_lengths = (double*) malloc(branch_count * sizeof(double));

	/* allocate branch length scalers if needed */
	if (brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED)
		treeinfo->brlen_scalers = (double*) calloc(partitions, sizeof(double));
	else
		treeinfo->brlen_scalers = NULL;

	/* check memory allocation */
	if (!treeinfo->partitions || !treeinfo->alphas || !treeinfo->param_indices || !treeinfo->subst_matrix_symmetries
			|| !treeinfo->branch_lengths || !treeinfo->deriv_precomp || !treeinfo->clv_valid || !treeinfo->pmatrix_valid
			|| !treeinfo->linked_branch_lengths || !treeinfo->partition_loglh || !treeinfo->gamma_mode
			|| !treeinfo->init_partition_idx
			|| (brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED && !treeinfo->brlen_scalers)) {
		throw std::runtime_error("Cannot allocate memory for treeinfo arrays\n");
		return NULL;
	}

	unsigned int p;
	for (p = 0; p < partitions; ++p) {
		/* use mean GAMMA rates per default */
		treeinfo->gamma_mode[p] = PLL_GAMMA_RATES_MEAN;

		/* allocate arrays for storing the per-partition branch lengths */
		if (brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
			treeinfo->branch_lengths[p] = (double*) malloc(branch_count * sizeof(double));
		} else
			treeinfo->branch_lengths[p] = treeinfo->linked_branch_lengths;

		/* initialize all branch length scalers to 1 */
		if (treeinfo->brlen_scalers)
			treeinfo->brlen_scalers[p] = 1.;

		/* check memory allocation */
		if (!treeinfo->branch_lengths[p]) {
			throw std::runtime_error("Cannot allocate memory for arrays for partition " + std::to_string(p));
			return NULL;
		}
	}

	/* by default, work with all partitions */
	treeinfo->active_partition = PLLMOD_TREEINFO_PARTITION_ALL;

	NetworkParams *params = (NetworkParams*) malloc(sizeof(NetworkParams));
	params->network = &network;
	params->network_treeinfo = treeinfo;
	treeinfo->likelihood_target_function = network_logl_wrapper;
	treeinfo->likelihood_computation_params = (void*) params;

	fake_init_collect_branch_lengths(treeinfo, network);

	return treeinfo;
}
void destroy_network_treeinfo(pllmod_treeinfo_t *treeinfo) {
	if (!treeinfo)
		return;
	if (treeinfo->likelihood_computation_params != treeinfo) {
		free(treeinfo->likelihood_computation_params);
	}
	pllmod_treeinfo_destroy(treeinfo);
}

TreeInfo createRaxmlTreeinfo(RaxmlInstance &instance, pllmod_treeinfo_t *treeinfo,
		TreeInfo::tinfo_behaviour &behaviour) {
	PartitionAssignment &part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());
	return TreeInfo(instance.opts, std::vector<doubleVector>(), treeinfo, (*instance.parted_msa.get()),
			instance.tip_msa_idmap, part_assign, std::vector<uintVector>(), behaviour);
}
pllmod_treeinfo_t* createStandardPllTreeinfo(const pll_utree_t *utree, unsigned int partitions, int brlen_linkage) {
	return pllmod_treeinfo_create(utree->vroot, utree->tip_count, partitions, brlen_linkage);
}

double network_logl_wrapper(void *network_params, int incremental, int update_pmatrices) {
	NetworkParams *params = (NetworkParams*) network_params;
	return computeLoglikelihood(*params->network, *params->network_treeinfo, incremental, update_pmatrices);
}
double network_opt_brlen_wrapper(pllmod_treeinfo_t *fake_treeinfo, double min_brlen, double max_brlen,
		double lh_epsilon, int max_iters, int opt_method, int radius) {
	Network *network = ((NetworkParams*) (fake_treeinfo->likelihood_computation_params))->network;
	return optimize_branches(*network, *fake_treeinfo, min_brlen, max_brlen, lh_epsilon, max_iters, opt_method, radius);

}
double network_spr_round_wrapper(pllmod_treeinfo_t *treeinfo, unsigned int radius_min, unsigned int radius_max,
		unsigned int ntopol_keep, pll_bool_t thorough, int brlen_opt_method, double bl_min, double bl_max,
		int smoothings, double epsilon, cutoff_info_t *cutoff_info, double subtree_cutoff) {
	throw std::runtime_error("Not implemented yet");
}
pllmod_ancestral_t* network_ancestral_wrapper(pllmod_treeinfo_t *treeinfo) {
	throw std::runtime_error("Not implemented yet");
}

}
