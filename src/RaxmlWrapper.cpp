/*
 * RaxmlWrapper.cpp
 *
 *  Created on: Sep 28, 2019
 *      Author: sarah
 */

#include "RaxmlWrapper.hpp"

#include "likelihood/LikelihoodComputation.hpp"
#include "optimization/BranchLengthOptimization.hpp"
#include "graph/Common.hpp"
#include "utils.hpp"
#include "graph/Network.hpp"
#include "graph/NetworkFunctions.hpp"

#include <raxml-ng/log.hpp>

namespace netrax {

void reset_tip_ids(Network &network, const std::unordered_map<std::string, size_t> &label_id_map) {
	if (label_id_map.size() < network.num_tips())
		throw std::invalid_argument("Invalid map size");

	for (auto &node : network.tip_nodes) {
		const unsigned int tip_id = label_id_map.at(node->label);
        node->clv_index = tip_id;
		node->getLink()->node_index = tip_id;
	}
}

void reset_tip_ids(pll_utree_t *utree, const std::unordered_map<std::string, size_t> &label_id_map) {
	for (size_t i = 0; i < utree->tip_count + utree->inner_count; ++i) {
		if (utree->nodes[i]->clv_index < utree->tip_count) {
			const unsigned int tip_id = label_id_map.at(utree->nodes[i]->label);
			pll_unode_t *node = utree->nodes[i];
			node->clv_index = node->node_index = tip_id;
		}
	}
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

RaxmlWrapper::RaxmlWrapper(const NetraxOptions &options) :
		netraxOptions(options) {
	instance = createRaxmlInstance(options);
	network_behaviour.compute_ancestral_function = [&](pllmod_treeinfo_t *treeinfo) {
		return network_ancestral_wrapper(treeinfo);
	};
	network_behaviour.opt_brlen_function = [&](pllmod_treeinfo_t *fake_treeinfo, double min_brlen, double max_brlen,
			double lh_epsilon, int max_iters, int opt_method, int radius) {
		return network_opt_brlen_wrapper(fake_treeinfo, min_brlen, max_brlen, lh_epsilon, max_iters, opt_method, radius);};
	network_behaviour.spr_round_function = [&](pllmod_treeinfo_t *treeinfo, unsigned int radius_min,
			unsigned int radius_max, unsigned int ntopol_keep, pll_bool_t thorough, int brlen_opt_method, double bl_min,
			double bl_max, int smoothings, double epsilon, cutoff_info_t *cutoff_info, double subtree_cutoff) {
		return network_spr_round_wrapper(treeinfo, radius_min, radius_max, ntopol_keep, thorough, brlen_opt_method,
				bl_min, bl_max, smoothings, epsilon, cutoff_info, subtree_cutoff);
	};
	network_behaviour.destroy_treeinfo_function = [&](pllmod_treeinfo_t *treeinfo) {
		return destroy_network_treeinfo(treeinfo);
	};
	network_behaviour.create_init_partition_function = network_create_init_partition_wrapper;
}

TreeInfo RaxmlWrapper::createRaxmlTreeinfo(Network &network) {
	// Check that the MSA has already been loaded
	assert(!instance.tip_id_map.empty());
	reset_tip_ids(network, instance.tip_id_map);
	assert(networkIsConnected(network));
	pllmod_treeinfo_t *pllTreeinfo = createNetworkPllTreeinfo(network, network.num_tips(),
			instance.parted_msa->part_count(), instance.opts.brlen_linkage);
	return createRaxmlTreeinfo(pllTreeinfo, network_behaviour);
}

pllmod_treeinfo_t* RaxmlWrapper::createStandardPllTreeinfo(const pll_utree_t *utree, unsigned int partitions,
		int brlen_linkage) {
	return pllmod_treeinfo_create(utree->vroot, utree->tip_count, partitions, brlen_linkage);
}

TreeInfo RaxmlWrapper::createRaxmlTreeinfo(pll_utree_t *utree) {
	// Check that the MSA has already been loaded
	assert(!instance.tip_id_map.empty());
	reset_tip_ids(utree, instance.tip_id_map);
	pllmod_treeinfo_t *pllTreeinfo = createStandardPllTreeinfo(utree, instance.parted_msa->part_count(),
			instance.opts.brlen_linkage);
	TreeInfo::tinfo_behaviour standard_behaviour;
	return createRaxmlTreeinfo(pllTreeinfo, standard_behaviour);
}

TreeInfo RaxmlWrapper::createRaxmlTreeinfo(pll_utree_t *utree, const pllmod_treeinfo_t &model_treeinfo) {
	// Check that the MSA has already been loaded
	assert(!instance.tip_id_map.empty());
	reset_tip_ids(utree, instance.tip_id_map);
	pllmod_treeinfo_t *pllTreeinfo = createStandardPllTreeinfo(utree, instance.parted_msa->part_count(),
			instance.opts.brlen_linkage);
	TreeInfo::tinfo_behaviour standard_behaviour;
	TreeInfo info = createRaxmlTreeinfo(pllTreeinfo, standard_behaviour);
	transfer_model_params(model_treeinfo, pllTreeinfo);
	return info;
}

size_t RaxmlWrapper::num_partitions() const {
	return instance.parted_msa->part_count();
}

void set_partition_fake_clv_entry(pll_partition_t *partition, size_t fake_clv_index) {
	unsigned int states = partition->states;
	unsigned int states_padded = partition->states_padded;
	unsigned int sites = partition->sites;
	unsigned int rate_cats = partition->rate_cats;

	// set clv to all-ones for the fake node
	double *clv = partition->clv[fake_clv_index];

	if (clv == NULL) { // this happens when we have site repeats
		// TODO: Does it work? Or do we need to increase the number of tips somehow when creating the partition?
		partition->clv[fake_clv_index] = (double*) pll_aligned_alloc(sites * rate_cats * states_padded * sizeof(double),
				partition->alignment);
		clv = partition->clv[fake_clv_index];
	}

	for (unsigned int n = 0; n < sites; ++n) {
		for (unsigned int i = 0; i < rate_cats; ++i) {
			for (unsigned int j = 0; j < states; ++j) {
				clv[j] = 1;
			}

			clv += states_padded;
		}
	}
}

int pllmod_treeinfo_init_partition_sarah(pllmod_treeinfo_t *treeinfo, unsigned int partition_index,
		pll_partition_t *partition, int params_to_optimize, int gamma_mode, double alpha,
		const unsigned int *param_indices, const int *subst_matrix_symmetries) {
	if (!treeinfo) {
		return PLL_FAILURE;
	} else if (partition_index >= treeinfo->partition_count) {
		return PLL_FAILURE;
	} else if (treeinfo->partitions[partition_index]) {
		return PLL_FAILURE;
	}

	unsigned int local_partition_index = treeinfo->init_partition_count++;
	treeinfo->partitions[partition_index] = partition;
	treeinfo->init_partitions[local_partition_index] = partition;
	treeinfo->init_partition_idx[local_partition_index] = partition_index;
	treeinfo->params_to_optimize[partition_index] = params_to_optimize;
	treeinfo->gamma_mode[partition_index] = gamma_mode;
	treeinfo->alphas[partition_index] = alpha;

	/* compute some derived dimensions, here is the only relevant change by sarah */
	unsigned int inner_nodes_count = treeinfo->tree->inner_count;
	unsigned int branch_count = treeinfo->tree->edge_count;
	unsigned int pmatrix_count = branch_count;
	unsigned int utree_count = inner_nodes_count * 3 + treeinfo->tip_count;

	/* allocate invalidation arrays */
	treeinfo->clv_valid[partition_index] = (char*) calloc(utree_count, sizeof(char));
	treeinfo->pmatrix_valid[partition_index] = (char*) calloc(pmatrix_count, sizeof(char));

	/* check memory allocation */
	if (!treeinfo->clv_valid[partition_index] || !treeinfo->pmatrix_valid[partition_index]) {
		return PLL_FAILURE;
	}

	/* allocate param_indices array and initialize it to all 0s,
	 * i.e. per default, all rate categories will use
	 * the same substitution matrix and same base frequencies */
	treeinfo->param_indices[partition_index] = (unsigned int*) calloc(partition->rate_cats, sizeof(unsigned int));

	/* check memory allocation */
	if (!treeinfo->param_indices[partition_index]) {
		return PLL_FAILURE;
	}

	/* if param_indices were provided, use them instead of default */
	if (param_indices)
		memcpy(treeinfo->param_indices[partition_index], param_indices, partition->rate_cats * sizeof(unsigned int));

	/* copy substitution rate matrix symmetries, if any */
	if (subst_matrix_symmetries) {
		const unsigned int symm_size = (partition->states * (partition->states - 1) / 2) * sizeof(int);
		treeinfo->subst_matrix_symmetries[partition_index] = (int*) malloc(symm_size);

		/* check memory allocation */
		if (!treeinfo->subst_matrix_symmetries[partition_index]) {
			return PLL_FAILURE;
		}

		memcpy(treeinfo->subst_matrix_symmetries[partition_index], subst_matrix_symmetries, symm_size);
	} else
		treeinfo->subst_matrix_symmetries[partition_index] = NULL;

	/* allocate memory for derivative precomputation table */
	unsigned int sites_alloc = partition->sites;
	if (partition->attributes & PLL_ATTRIB_AB_FLAG)
		sites_alloc += partition->states;
	unsigned int precomp_size = sites_alloc * partition->rate_cats * partition->states_padded;

	treeinfo->deriv_precomp[partition_index] = (double*) pll_aligned_alloc(precomp_size * sizeof(double),
			partition->alignment);

	if (!treeinfo->deriv_precomp[partition_index]) {
		return PLL_FAILURE;
	}

	memset(treeinfo->deriv_precomp[partition_index], 0, precomp_size * sizeof(double));

	return PLL_SUCCESS;
}

void RaxmlWrapper::network_create_init_partition_wrapper(size_t p, int params_to_optimize,
		pllmod_treeinfo_t *pll_treeinfo, const Options &opts, const PartitionInfo &pinfo, const IDVector &tip_msa_idmap,
		PartitionAssignment::const_iterator &part_range, const uintVector &weights) {
	/* create and init PLL partition structure */
	pll_partition_t *partition = create_pll_partition(opts, pinfo, tip_msa_idmap, *part_range, weights,
			pll_treeinfo->tree->tip_count, pll_treeinfo->tree->inner_count, pll_treeinfo->tree->edge_count);
	int retval = pllmod_treeinfo_init_partition_sarah(pll_treeinfo, p, partition, params_to_optimize,
			pinfo.model().gamma_mode(), pinfo.model().alpha(), pinfo.model().ratecat_submodels().data(),
			pinfo.model().submodel(0).rate_sym().data());
	if (!retval) {
		assert(pll_errno);
		libpll_check_error("ERROR adding treeinfo partition");
	}
	set_partition_fake_clv_entry(partition, pll_treeinfo->tree->tip_count + pll_treeinfo->tree->inner_count - 1);
}

void RaxmlWrapper::network_init_treeinfo_wrapper(const Options &opts, const std::vector<doubleVector> &partition_brlens,
		size_t num_branches, const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
		const PartitionAssignment &part_assign, const std::vector<uintVector> &site_weights,
		doubleVector *partition_contributions, pllmod_treeinfo_t *pll_treeinfo, IDSet *parts_master) {

	//throw std::runtime_error("not implemented yet");
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
			set_partition_fake_clv_entry(partition, pll_treeinfo->tree->inner_count);

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
		treeinfo->branch_lengths[0][i] = network.edges[i].length;
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

pllmod_treeinfo_t* RaxmlWrapper::createNetworkPllTreeinfo(Network &network, unsigned int tips, unsigned int partitions,
		int brlen_linkage) {
	assert(partitions > 0);

	if (partitions > 1) {
		for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
			network.reticulation_nodes[i]->getReticulationData()->setNumPartitions(partitions);
		}
	}

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
	params->raxml_wrapper = this;
	treeinfo->likelihood_target_function = network_logl_wrapper;
	treeinfo->likelihood_computation_params = (void*) params;

	fake_init_collect_branch_lengths(treeinfo, network);

	return treeinfo;
}

void RaxmlWrapper::destroy_network_treeinfo(pllmod_treeinfo_t *treeinfo) {
	if (!treeinfo)
		return;
	if (treeinfo->likelihood_computation_params != treeinfo) {
		free(treeinfo->likelihood_computation_params);
	}
	pllmod_treeinfo_destroy(treeinfo);
}

TreeInfo RaxmlWrapper::createRaxmlTreeinfo(pllmod_treeinfo_t *treeinfo, TreeInfo::tinfo_behaviour &behaviour) {
	PartitionAssignment &part_assign = instance.proc_part_assign.at(ParallelContext::proc_id());
	return TreeInfo(instance.opts, std::vector<doubleVector>(), treeinfo, (*instance.parted_msa.get()),
			instance.tip_msa_idmap, part_assign, std::vector<uintVector>(), behaviour);
}

double RaxmlWrapper::network_logl_wrapper(void *network_params, int incremental, int update_pmatrices) {
	NetworkParams *params = (NetworkParams*) network_params;
	return computeLoglikelihood(*params->network, *params->network_treeinfo, incremental,
			update_pmatrices);
}
double RaxmlWrapper::network_opt_brlen_wrapper(pllmod_treeinfo_t *fake_treeinfo, double min_brlen, double max_brlen,
		double lh_epsilon, int max_iters, int opt_method, int radius) {
	Network *network = ((NetworkParams*) (fake_treeinfo->likelihood_computation_params))->network;
	return optimize_branches(netraxOptions, *network, *fake_treeinfo, min_brlen, max_brlen, lh_epsilon, max_iters,
			opt_method, radius);

}
double RaxmlWrapper::network_spr_round_wrapper(pllmod_treeinfo_t *treeinfo, unsigned int radius_min,
		unsigned int radius_max, unsigned int ntopol_keep, pll_bool_t thorough, int brlen_opt_method, double bl_min,
		double bl_max, int smoothings, double epsilon, cutoff_info_t *cutoff_info, double subtree_cutoff) {
	(void)treeinfo;
	(void)radius_min;
	(void)radius_max;
	(void)ntopol_keep;
	(void)thorough;
	(void)brlen_opt_method;
	(void)bl_min;
	(void)bl_max;
	(void)smoothings;
	(void)epsilon;
	(void)cutoff_info;
	(void)subtree_cutoff;
	throw std::runtime_error("Not implemented yet");
}
pllmod_ancestral_t* RaxmlWrapper::network_ancestral_wrapper(pllmod_treeinfo_t *treeinfo) {
	(void)treeinfo;
	throw std::runtime_error("Not implemented yet");
}

Options RaxmlWrapper::getRaxmlOptions() const {
	return instance.opts;
}

void RaxmlWrapper::enableRaxmlDebugOutput() {
	instance.opts.log_level = LogLevel::debug;
	logger().log_level(instance.opts.log_level);
	logger().add_log_stream(&cout);
}

}
