/*
 * RaxmlWrapper.hpp
 *
 *  Created on: Sep 28, 2019
 *      Author: sarah
 */

#pragma once

#include <raxml-ng/main.hpp>
#include <raxml-ng/Model.hpp>

#include "NetraxOptions.hpp"
#include "graph/Network.hpp"
#include "graph/AnnotatedNetwork.hpp"

namespace netrax {

class RaxmlWrapper {
public:
    struct NetworkParams {
        AnnotatedNetwork *ann_network;
        NetworkParams(AnnotatedNetwork *ann_network) :
                ann_network(ann_network) {
        }
    };

    RaxmlWrapper(NetraxOptions &options);

    Options getRaxmlOptions() const;

    TreeInfo* createRaxmlTreeinfo(AnnotatedNetwork &ann_network); // Creates a network treeinfo
    TreeInfo* createRaxmlTreeinfo(pll_utree_t *utree); // Creates a tree treeinfo
    TreeInfo* createRaxmlTreeinfo(pll_utree_t *utree, const pllmod_treeinfo_t &model_treeinfo); // Creates a tree treeinfo, taking the model from the given treeinfo

    // and now, the things only neccessary to be visible in this header because of the unit tests...

    TreeInfo* createRaxmlTreeinfo(pllmod_treeinfo_t *treeinfo,
            TreeInfo::tinfo_behaviour &behaviour);
    pllmod_treeinfo_t* createStandardPllTreeinfo(const pll_utree_t *utree, unsigned int partitions,
            int brlen_linkage);
    pllmod_treeinfo_t* createNetworkPllTreeinfoInternal(AnnotatedNetwork &ann_network, unsigned int tips,
            unsigned int partitions, int brlen_linkage);
    pllmod_treeinfo_t* createNetworkPllTreeinfo(AnnotatedNetwork &ann_network);
    //void destroy_network_treeinfo(pllmod_treeinfo_t *treeinfo);

    void enableRaxmlDebugOutput();

    static double network_logl_wrapper(void *network_params, int incremental, int update_pmatrices, double ** persite_lnl);
    double network_opt_brlen_wrapper(pllmod_treeinfo_t *fake_treeinfo, double min_brlen,
            double max_brlen, double lh_epsilon, int max_iters, int opt_method, int radius);
    double network_spr_round_wrapper(pllmod_treeinfo_t *treeinfo, unsigned int radius_min,
            unsigned int radius_max, unsigned int ntopol_keep, pll_bool_t thorough,
            int brlen_opt_method, double bl_min, double bl_max, int smoothings, double epsilon,
            cutoff_info_t *cutoff_info, double subtree_cutoff);
    pllmod_ancestral_t* network_ancestral_wrapper(pllmod_treeinfo_t *treeinfo);
    void network_init_treeinfo_wrapper(const Options &opts,
            const std::vector<doubleVector> &partition_brlens, size_t num_branches,
            const PartitionedMSA &parted_msa, const IDVector &tip_msa_idmap,
            const PartitionAssignment &part_assign, const std::vector<uintVector> &site_weights,
            doubleVector *partition_contributions, pllmod_treeinfo_t *pll_treeinfo,
            IDSet *parts_master);
    static void network_create_init_partition_wrapper(size_t p, int params_to_optimize,
            pllmod_treeinfo_t *pll_treeinfo, const Options &opts, const PartitionInfo &pinfo,
            const IDVector &tip_msa_idmap, PartitionAssignment::const_iterator &part_range,
            const uintVector &weights);

    size_t num_partitions() const;

    Tree generateRandomTree(double seed);
    Tree generateParsimonyTree(double seed);
    Tree bestRaxmlTree() const;

private:
    RaxmlInstance instance;
    TreeInfo::tinfo_behaviour network_behaviour;
    NetraxOptions netraxOptions;
};

}
