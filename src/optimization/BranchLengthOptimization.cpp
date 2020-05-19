/*
 * BranchLengthOptimization.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: Sarah Lutteropp
 */

#include "BranchLengthOptimization.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>

#include "../graph/Common.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "../RaxmlWrapper.hpp"
#include "../utils.hpp"
#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

struct OptimizedBranchLength {
    size_t tree_index;
    double length;
    double tree_prob;
};

const static bool DO_BRLEN_VARIANCE_EXPERIMENT = true;

double computeVariance(const std::vector<OptimizedBranchLength> &brlens) {
    // TODO: Maybe adapt to different tree probabilities, such as here?
    // https://en.wikipedia.org/wiki/Variance#Discrete_random_variable
    double var = 0;
    if (brlens.empty()) {
        var = std::numeric_limits<double>::infinity();
    } else {
        // compute the mean branch length
        double mean = 0;
        for (size_t i = 0; i < brlens.size(); ++i) {
            mean += brlens[i].length;
        }
        mean /= (double) brlens.size();

        // compute the variance
        for (size_t i = 0; i < brlens.size(); ++i) {
            var += (brlens[i].length - mean) * (brlens[i].length - mean);
        }
        var /= (double) brlens.size();
    }
    return var;
}

double optimize_branches(AnnotatedNetwork &ann_network, double min_brlen, double max_brlen, double lh_epsilon,
        int max_iters, int opt_method, int radius) {
    NetraxOptions &options = ann_network.options;
    Network &network = ann_network.network;
    pllmod_treeinfo_t &fake_treeinfo = *ann_network.fake_treeinfo;
    bool unlinked_mode = (fake_treeinfo.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);

    // for now, optimize branches on each of the displayed trees, exported as a pll_utree_t data structure.
    // Keep track of which branch in the exported pll_utree_t corresponds to which branch of the network data structure.
    // Run br-length optimization on each of the displayed trees.
    // For each branch in the network, collect the optimized branch lengths from the displayed trees.
    // Print the optimized displayed tree branch lengths (as well as displayed tree prob) for each branch in the network to a file.
    // for now, set the network brlens to a weighted average of the displayed-tree-brlens (weighted by tree probability)
    // also print the new network brlens.
    // Do some plots.

    size_t partitionIdx = 0;

    std::vector<double> old_brlens(network.num_branches());
    for (size_t i = 0; i < old_brlens.size(); ++i) {
        old_brlens[network.edges[i].pmatrix_index] = network.edges[i].length;
    }

    std::vector<std::vector<OptimizedBranchLength> > opt_brlens(network.num_branches());
    size_t n_trees = 1 << network.num_reticulations();

    for (size_t tree_idx = 0; tree_idx < n_trees; tree_idx++) {
        if (displayed_tree_prob(ann_network, tree_idx, unlinked_mode ? 0 : partitionIdx) == 0.0) {
            continue;
        }

        setReticulationParents(network, tree_idx);
        pll_utree_t *displayed_utree = displayed_tree_to_utree(network, ann_network.travbuffer, tree_idx);
        std::vector<std::vector<size_t> > dtBranchToNetworkBranch = getDtBranchToNetworkBranchMapping(*displayed_utree,
                network, tree_idx);

        // optimize brlens on the tree
        NetraxOptions opts;
        RaxmlWrapper wrapper(options);
        TreeInfo *tInfo = wrapper.createRaxmlTreeinfo(displayed_utree, fake_treeinfo);
        Options raxmlOptions = wrapper.getRaxmlOptions();

        // TODO: Remove this again, it was only here because of the Slack discussion
        /*tInfo.optimize_model(raxmlOptions.lh_epsilon);
         std::cout << "displayed tree #" << tree_idx << " would like these model params:\n";
         const pll_partition_t* partition = tInfo.pll_treeinfo().partitions[0];
         print_model_params(*partition);
         std::cout << "\n";*/

        tInfo->optimize_branches(raxmlOptions.lh_epsilon, 0.25);
        const pllmod_treeinfo_t &pllmod_tInfo = tInfo->pll_treeinfo();

        // collect optimized brlens from the tree
        for (size_t i = 0; i < pllmod_tInfo.tree->edge_count; ++i) {
            if (dtBranchToNetworkBranch[i].size() == 1) {
                size_t networkBranchIdx = dtBranchToNetworkBranch[i][0];
                double new_brlen = pllmod_tInfo.branch_lengths[partitionIdx][i];
                double tree_prob = displayed_tree_prob(ann_network, tree_idx, unlinked_mode ? 0 : partitionIdx);
                opt_brlens[networkBranchIdx].push_back(OptimizedBranchLength { tree_idx, new_brlen, tree_prob });
            }
        }

        delete tInfo;

        // print the displayed tree as NEWICK:
        /*std::cout << "displayed tree #" << tree_idx << " as NEWICK, after brlen opt on this tree:\n";
         char *text = pll_utree_export_newick(displayed_utree->vroot, NULL);
         std::string str(text);
         std::cout << str << "\n";
         free(text);*/

    }

    std::vector<bool> visited(network.nodes.size(), false);
    // set the network brlens to the weighted average of the displayed_tree brlens
    // also set the network brlen support values to the brlen variance in the displayed trees which are having this branch
    for (size_t i = 0; i < network.num_branches(); ++i) {
        size_t networkBranchIdx = network.edges[i].pmatrix_index;
        if (!opt_brlens[networkBranchIdx].empty()) {
            double treeProbSum = 0;
            for (size_t j = 0; j < opt_brlens[networkBranchIdx].size(); ++j) {
                treeProbSum += opt_brlens[networkBranchIdx][j].tree_prob;
            }
            assert(treeProbSum != 0);
            double newLength = 0;
            for (size_t j = 0; j < opt_brlens[networkBranchIdx].size(); ++j) {
                double weight = opt_brlens[networkBranchIdx][j].tree_prob / treeProbSum;
                newLength += opt_brlens[networkBranchIdx][j].length * weight;
            }
            if (network.edges[i].length != newLength) {
                invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, getSource(network, &network.edges[i]),
                        false);
            }
            network.edges[i].length = newLength;
            fake_treeinfo.branch_lengths[partitionIdx][networkBranchIdx] = newLength;
        }
        network.edges[i].support = computeVariance(opt_brlens[networkBranchIdx]);
    }

    if (DO_BRLEN_VARIANCE_EXPERIMENT) {
        // for each network branch length, do the printing
        std::cout << std::setprecision(17) << "\n";
        for (size_t i = 0; i < network.num_branches(); ++i) {
            size_t networkBranchIdx = network.edges[i].pmatrix_index;
            std::cout << "Network branch " << networkBranchIdx << ":\n";
            if (!opt_brlens[networkBranchIdx].empty()) {
                std::cout << " Old brlen before optimization: " << old_brlens[networkBranchIdx] << "\n";
                std::cout << " New brlen from weighted average: " << network.edges[networkBranchIdx].length << "\n";
                for (size_t j = 0; j < opt_brlens[i].size(); ++j) {
                    std::cout << "  Tree #" << opt_brlens[networkBranchIdx][j].tree_index << ", prob = "
                            << opt_brlens[networkBranchIdx][j].tree_prob << ", opt_brlen = "
                            << opt_brlens[networkBranchIdx][j].length << "\n";
                }
                assert(network.edges[i].length == fake_treeinfo.branch_lengths[partitionIdx][networkBranchIdx]);
            } else {
                std::cout << " This branch is has no exact presence in any displayed tree.\n";
            }
        }

        // just for debug: printing all network branch lengths for the partition 0
        std::cout << "End of BRLEN_OPT function - All network branch lengths for partition 0:\n";
        for (size_t i = 0; i < network.num_branches(); ++i) {
            std::cout << " pmatrix_idx = " << network.edges[i].pmatrix_index << " -> brlen = "
                    << network.edges_by_index[network.edges[i].pmatrix_index]->length << "\n";
        }
    }

    return -1 * computeLoglikelihood(ann_network, 1, 1, false);
}

}
