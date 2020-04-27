/*
 * Api.cpp
 *
 *  Created on: Apr 24, 2020
 *      Author: sarah
 */

#include "Api.hpp"

#include <stddef.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <raxml-ng/TreeInfo.hpp>
#include "graph/AnnotatedNetwork.hpp"
#include "graph/BiconnectedComponents.hpp"
#include "graph/Network.hpp"
#include "graph/NetworkFunctions.hpp"
#include "graph/NetworkTopology.hpp"
#include "io/NetworkIO.hpp"
#include "likelihood/LikelihoodComputation.hpp"
#include "NetraxOptions.hpp"
#include "RaxmlWrapper.hpp"

namespace netrax {

AnnotatedNetwork build_annotated_network(const NetraxOptions &options) {
    AnnotatedNetwork ann_network;
    ann_network.network = netrax::readNetworkFromFile(options.network_file, options.max_reticulations);
    Network &network = ann_network.network;
    ann_network.options = options;

    netrax::RaxmlWrapper wrapper(options);
    ann_network.raxml_treeinfo = std::unique_ptr<TreeInfo>(wrapper.createRaxmlTreeinfo(ann_network));
    ann_network.blobInfo = netrax::partitionNetworkIntoBlobs(ann_network.network);

    // init branch probs...
    if (options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) { // common branches
        ann_network.branch_probs = std::vector<std::vector<double> >(1,
                std::vector<double>(ann_network.network.edges.size() + 1, 1.0));
    } else { // each partition has extra branch properties
        ann_network.branch_probs = std::vector<std::vector<double> >(ann_network.fake_treeinfo->partition_count,
                std::vector<double>(ann_network.network.edges.size() + 1, 1.0));
    }
    for (size_t p = 0; p < ann_network.branch_probs.size(); ++p) {
        for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
            Node *retNode = ann_network.network.reticulation_nodes[i];
            double firstParentProb = netrax::getReticulationFirstParentProb(network, retNode);
            double secondParentProb = netrax::getReticulationSecondParentProb(network, retNode);
            size_t firstParentEdgeIndex = netrax::getReticulationFirstParentPmatrixIndex(network, retNode);
            size_t secondParentEdgeIndex = netrax::getReticulationSecondParentPmatrixIndex(network, retNode);
            ann_network.branch_probs[p][firstParentEdgeIndex] = firstParentProb;
            ann_network.branch_probs[p][secondParentEdgeIndex] = secondParentProb;
        }
    }
    return ann_network;
}

double computeLoglikelihood(AnnotatedNetwork &ann_network) {
    return ann_network.raxml_treeinfo->loglh(false);
}
double updateReticulationProbs(AnnotatedNetwork &ann_network) {
    return netrax::computeLoglikelihood(ann_network, 0, 1, true);
}
double optimizeModel(AnnotatedNetwork &ann_network) {
    return ann_network.raxml_treeinfo->optimize_model(ann_network.options.lh_epsilon);
}
double optimizeBranches(AnnotatedNetwork &ann_network) {
    return ann_network.raxml_treeinfo->optimize_branches(ann_network.options.lh_epsilon, 1);
}
double optimizeTopology(AnnotatedNetwork &ann_network) {
    std::cout << "(Topology optimization not implemented yet) \n";
    return -1;
}

void writeNetwork(AnnotatedNetwork &ann_network, const std::string &filepath) {
    std::ofstream outfile(filepath);
    // If we have unlinked branch lenghts/probs, replace the entries in the network by their average
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED
            && ann_network.fake_treeinfo->partition_count > 1) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            double lenSum = 0.0;
            double probSum = 0.0;
            size_t pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                lenSum += ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
                probSum += ann_network.branch_probs[p][pmatrix_index];
            }
            ann_network.network.edges[i].length = lenSum / ann_network.fake_treeinfo->partition_count;
            ann_network.network.edges[i].prob = probSum / ann_network.fake_treeinfo->partition_count;
        }
    }

    outfile << netrax::toExtendedNewick(ann_network.network) << "\n";
    outfile.close();
}

}
