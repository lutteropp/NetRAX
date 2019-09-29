/*
 * RaxmlWrapper.hpp
 *
 *  Created on: Sep 28, 2019
 *      Author: sarah
 */

#pragma once

#include <raxml-ng/main.hpp>

#include "NetraxOptions.hpp"
#include "Network.hpp"

namespace netrax {

class RaxmlWrapper {
public:
	RaxmlWrapper(const NetraxOptions &options);

	TreeInfo createRaxmlTreeinfo(Network &network); // Creates a network treeinfo
	TreeInfo createRaxmlTreeinfo(const pll_utree_t *utree); // Creates a tree treeinfo

	// and now, the things only neccessary to be visible in this header because of the unit tests...

	TreeInfo createRaxmlTreeinfo(pllmod_treeinfo_t *treeinfo, TreeInfo::tinfo_behaviour &behaviour);
	pllmod_treeinfo_t* createStandardPllTreeinfo(const pll_utree_t *utree, unsigned int partitions, int brlen_linkage);
	pllmod_treeinfo_t* createNetworkPllTreeinfo(Network &network, unsigned int tips, unsigned int partitions,
			int brlen_linkage);
	void destroy_network_treeinfo(pllmod_treeinfo_t *treeinfo);

	struct NetworkParams {
		Network *network;
		pllmod_treeinfo_t *network_treeinfo;
		RaxmlWrapper *raxml_wrapper;
		NetworkParams(Network *network, pllmod_treeinfo_t *network_treeinfo, RaxmlWrapper *raxml_wrapper) :
				network(network), network_treeinfo(network_treeinfo), raxml_wrapper(raxml_wrapper) {
		}
	};

	static double network_logl_wrapper(void *network_params, int incremental, int update_pmatrices);
	double network_opt_brlen_wrapper(pllmod_treeinfo_t *fake_treeinfo, double min_brlen, double max_brlen,
			double lh_epsilon, int max_iters, int opt_method, int radius);
	double network_spr_round_wrapper(pllmod_treeinfo_t *treeinfo, unsigned int radius_min, unsigned int radius_max,
			unsigned int ntopol_keep, pll_bool_t thorough, int brlen_opt_method, double bl_min, double bl_max,
			int smoothings, double epsilon, cutoff_info_t *cutoff_info, double subtree_cutoff);
	pllmod_ancestral_t* network_ancestral_wrapper(pllmod_treeinfo_t *treeinfo);
	void network_init_treeinfo_wrapper(const Options &opts, const std::vector<doubleVector> &partition_brlens,
			size_t num_branches, const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
			const PartitionAssignment &part_assign, const std::vector<uintVector> &site_weights,
			doubleVector *partition_contributions, pllmod_treeinfo_t *pll_treeinfo, IDSet *parts_master);
private:
	pllmod_treeinfo_t* createNetworkPllTreeinfo_new(Network &network, unsigned int tips, unsigned int partitions,
			int brlen_linkage);
	pllmod_treeinfo_t* createNetworkPllTreeinfo_buggy(Network &network, unsigned int tips, unsigned int partitions,
			int brlen_linkage);

	RaxmlInstance instance;
	TreeInfo::tinfo_behaviour network_behaviour;
};

}
