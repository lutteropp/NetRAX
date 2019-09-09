/*
 * Fake.cpp
 *
 *  Created on: Sep 9, 2019
 *      Author: Sarah Lutteropp
 */

#include "Fake.hpp"

#include "likelihood/LikelihoodComputation.hpp"

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

int fake_init_tree(pllmod_treeinfo_t * treeinfo, Network& network) {
	pll_utree_t * tree = (pll_utree_t*) malloc(sizeof(pll_utree_t));
	treeinfo->tree = tree;

	tree->tip_count = network.tip_count + 1; // +1 for the fake clv index, TODO: Do we need it here?
	tree->edge_count = network.edges.size() + 1; // +1 for the fake pmatrix index, TODO: Do we need it here?
	tree->inner_count = network.nodes.size();

	treeinfo->root = NULL;

	// collect the branch lengths
	for (size_t i = 0; i < network.edges.size(); ++i)  {
		treeinfo->branch_lengths[0][i] = network.edges[i].getLength();
	}
	treeinfo->branch_lengths[0][network.edges.size()] = 0; // the fake branch length

	/* in unlinked branch length mode, copy brlen to other partitions */
	if (treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
		for (unsigned int i = 1; i < treeinfo->partition_count; ++i) {
			// TODO: only save brlens for initialized partitions
//      if (treeinfo->partitions[i])
			{
				memcpy(treeinfo->branch_lengths[i], treeinfo->branch_lengths[0], tree->edge_count * sizeof(double));
			}
		}
	}

	return PLL_SUCCESS;
}

void destroy_fake_treeinfo(pllmod_treeinfo_t * treeinfo) {
	if (!treeinfo)
		return;

	/* deallocate traversal buffer, branch lengths array, matrix indices
	 array and operations */
	//free(treeinfo->travbuffer);
	//free(treeinfo->matrix_indices);
	//free(treeinfo->operations);
	//free(treeinfo->subnodes);
	/* destroy all structures allocated for the concrete PLL partition instance */
	unsigned int p;
	for (p = 0; p < treeinfo->partition_count; ++p) {
		if (treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED)
			free(treeinfo->branch_lengths[p]);

		pllmod_treeinfo_destroy_partition(treeinfo, p);
	}

	if (treeinfo->subst_matrix_symmetries)
		free(treeinfo->subst_matrix_symmetries);

	if (treeinfo->constraint)
		free(treeinfo->constraint);

	/* free invalidation arrays */
	free(treeinfo->clv_valid);
	free(treeinfo->pmatrix_valid);

	free(treeinfo->linked_branch_lengths);

	/* free alpha and param_indices arrays */
	free(treeinfo->params_to_optimize);
	free(treeinfo->alphas);
	free(treeinfo->gamma_mode);
	free(treeinfo->param_indices);
	free(treeinfo->branch_lengths);
	free(treeinfo->partition_loglh);
	free(treeinfo->deriv_precomp);

	if (treeinfo->brlen_scalers)
		free(treeinfo->brlen_scalers);

	/* deallocate partition array */
	free(treeinfo->partitions);
	free(treeinfo->init_partitions);
	free(treeinfo->init_partition_idx);

	if (treeinfo->tree) {
		//free(treeinfo->tree->nodes);
		free(treeinfo->tree);
	}

	free(treeinfo->likelihood_computation_params);

	/* finally, deallocate treeinfo object itself */
	free(treeinfo);
}

double fake_network_loglikelihood(void* network_params, int incremental, int update_pmatrices) {
	NetworkParams* params = (NetworkParams*) network_params;
	return computeLoglikelihood(*params->network, *params->fake_treeinfo);
}

pllmod_treeinfo_t * create_fake_treeinfo(Network& network, unsigned int tips, unsigned int partitions, int brlen_linkage) {
	/* create treeinfo instance */
	pllmod_treeinfo_t * treeinfo;

	if (!(treeinfo = (pllmod_treeinfo_t *) calloc(1, sizeof(pllmod_treeinfo_t)))) {
		throw std::runtime_error("Cannot allocate memory for treeinfo\n");
		return NULL;
	}

	/* save dimensions & options */
	treeinfo->tip_count = tips;
	treeinfo->partition_count = partitions;
	treeinfo->brlen_linkage = brlen_linkage;

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
	treeinfo->partitions = (pll_partition_t **) calloc(partitions, sizeof(pll_partition_t *));
	treeinfo->params_to_optimize = (int *) calloc(partitions, sizeof(int));
	treeinfo->alphas = (double *) calloc(partitions, sizeof(double));
	treeinfo->gamma_mode = (int *) calloc(partitions, sizeof(int));
	treeinfo->param_indices = (unsigned int **) calloc(partitions, sizeof(unsigned int*));
	treeinfo->subst_matrix_symmetries = (int **) calloc(partitions, sizeof(int*));
	treeinfo->branch_lengths = (double **) calloc(partitions, sizeof(double*));
	treeinfo->deriv_precomp = (double **) calloc(partitions, sizeof(double*));
	treeinfo->clv_valid = (char **) calloc(partitions, sizeof(char*));
	treeinfo->pmatrix_valid = (char **) calloc(partitions, sizeof(char*));
	treeinfo->partition_loglh = (double *) calloc(partitions, sizeof(double));

	treeinfo->init_partition_count = 0;
	treeinfo->init_partition_idx = (unsigned int *) calloc(partitions, sizeof(unsigned int));
	treeinfo->init_partitions = (pll_partition_t **) calloc(partitions, sizeof(pll_partition_t *));

	/* allocate array for storing linked/average branch lengths */
	treeinfo->linked_branch_lengths = (double *) malloc(branch_count * sizeof(double));

	/* allocate branch length scalers if needed */
	if (brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED)
		treeinfo->brlen_scalers = (double *) calloc(partitions, sizeof(double));
	else
		treeinfo->brlen_scalers = NULL;

	/* check memory allocation */
	if (!treeinfo->partitions || !treeinfo->alphas || !treeinfo->param_indices || !treeinfo->subst_matrix_symmetries
			|| !treeinfo->branch_lengths || !treeinfo->deriv_precomp || !treeinfo->clv_valid || !treeinfo->pmatrix_valid
			|| !treeinfo->linked_branch_lengths || !treeinfo->partition_loglh || !treeinfo->gamma_mode || !treeinfo->init_partition_idx
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
			treeinfo->branch_lengths[p] = (double *) malloc(branch_count * sizeof(double));
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

	fake_init_tree(treeinfo, network);

	NetworkParams* params = (NetworkParams*) malloc(sizeof(NetworkParams));
	params->network = &network;
	params->fake_treeinfo = treeinfo;
	treeinfo->likelihood_target_function = fake_network_loglikelihood;
	treeinfo->likelihood_computation_params = (void*) params;

	return treeinfo;
}

}
