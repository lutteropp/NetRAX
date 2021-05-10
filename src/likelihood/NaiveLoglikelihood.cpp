#include "NaiveLoglikelihood.hpp"

#include "../RaxmlWrapper.hpp"
#include "mpreal.h"
#include "../helper/NetworkTopology.hpp"
#include "../helper/NetworkFunctions.hpp"

namespace netrax {

double displayed_tree_logprob(AnnotatedNetwork &ann_network, size_t tree_index) {
    Network &network = ann_network.network;
    setReticulationParents(network, tree_index);
    mpfr::mpreal logProb = 0;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        mpfr::mpreal prob = getReticulationActiveProb(ann_network, network.reticulation_nodes[i]);
        if (prob == 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        logProb += mpfr::log(prob);
    }
    return logProb.toDouble();
}

/**
 * Compute network loglikelihood by converting each displayed tree into a pll_utree_t and calling raxml loglikelihood on it.
 * Then taking the weighted sum of the displayed tree likelihoods (using exp) for the network likelihood, and returning th log of the network likelihood.
 * Using arbitrary-precision floating point operations for aggregating over the displayed tree loglikelihoods.
 * This naive implementation assumes a single partition in the MSA. It also doesn't do model optimization.
 **/
double computeLoglikelihoodNaiveUtree(AnnotatedNetwork &ann_network, int incremental,
        int update_pmatrices, std::vector<double> *treewise_logl) {
    (void) incremental;
    (void) update_pmatrices;
    Network &network = ann_network.network;
    size_t num_partitions = ann_network.fake_treeinfo->partition_count;
    size_t num_trees = (1 << ann_network.network.num_reticulations());

    std::vector<Model> partition_models(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
        assign(partition_models[i], ann_network.fake_treeinfo->partitions[i]);
    }

    assert(ann_network.options.brlen_linkage != PLLMOD_COMMON_BRLEN_UNLINKED);
    // ensure the network edge lengths are up-to-date (needed for displayed tree to utree)
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        ann_network.network.edges_by_index[i]->length = ann_network.fake_treeinfo->branch_lengths[0][i];
    }

    std::vector<std::vector<double> > tree_logl_per_partition(num_partitions, std::vector<double>(num_trees, -std::numeric_limits<double>::infinity()));
    std::vector<double> tree_logprob(num_trees, 0.0);

    // Iterate over all displayed trees
    for (size_t i = 0; i < num_trees; ++i) {
        tree_logprob[i] =  displayed_tree_logprob(ann_network, i);
        if (tree_logprob[i] == std::numeric_limits<double>::infinity()) {
            continue;
        }
        pll_utree_t *displayed_tree = netrax::displayed_tree_to_utree(network, i);
        TreeInfo *displayedTreeinfo = createRaxmlTreeinfo(displayed_tree, ann_network.instance);
        for (size_t p = 0; p < num_partitions; ++p) {
            displayedTreeinfo->model(p, partition_models[p]);
        }

        double tree_logl = displayedTreeinfo->loglh(0);
        for (size_t p = 0; p < num_partitions; ++p) {
            tree_logl_per_partition[p][i] = displayedTreeinfo->pll_treeinfo().partition_loglh[p];
        }
        delete displayedTreeinfo;

        if (treewise_logl) {
            treewise_logl->emplace_back(tree_logl);
        }
        assert(tree_logl != -std::numeric_limits<double>::infinity());
    }

    mpfr::mpreal network_logl = 0.0;
    for (size_t partition_idx = 0; partition_idx < num_partitions; ++partition_idx) {
        double partition_logl = -std::numeric_limits<double>::infinity();
        if (ann_network.options.likelihood_variant == LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
            mpfr::mpreal partition_lh = 0.0;
            for (size_t i = 0; i < num_trees; ++i) {
                if (tree_logprob[i] != std::numeric_limits<double>::infinity()) {
                    partition_lh += mpfr::exp(tree_logl_per_partition[partition_idx][i]) * mpfr::exp(tree_logprob[i]);
                }
            }
            partition_logl = mpfr::log(partition_lh).toDouble();
        } else { // LikelihoodVariant::BEST_DISPLAYED_TREE
            for (size_t i = 0; i < num_trees; ++i) {
                if (tree_logprob[i] != std::numeric_limits<double>::infinity()) {
                    partition_logl = std::max(partition_logl, tree_logl_per_partition[partition_idx][i] + tree_logprob[i]);
                }
            }
            //std::cout << "partiion " << partition_idx << " logl: " << partition_logl << "\n";
        }
        network_logl += partition_logl;
        ann_network.fake_treeinfo->partition_loglh[partition_idx] = partition_logl;
    }
    return network_logl.toDouble();
}

}