/*
 * Api.cpp
 *
 *  Created on: Apr 24, 2020
 *      Author: sarah
 */

#include "Api.hpp"

#include <stddef.h>
#include <iostream>
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

    ann_network.network = netrax::readNetworkFromFile(options.network_file);
    ann_network.options = options;

    netrax::RaxmlWrapper wrapper(options);
    ann_network.raxml_treeinfo = std::unique_ptr<TreeInfo>(wrapper.createRaxmlTreeinfo(ann_network));
    ann_network.blobInfo = netrax::partitionNetworkIntoBlobs(ann_network.network);

    // init branch probs...
    if (options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) { // common branches
        ann_network.branch_probs = std::vector<std::vector<double> >(1,
                std::vector<double>(ann_network.network.num_edges(), 1.0));
    } else { // each partition has extra branch properties
        ann_network.branch_probs = std::vector<std::vector<double> >(ann_network.fake_treeinfo->partition_count,
                std::vector<double>(ann_network.network.num_edges(), 1.0));
    }
    for (size_t p = 0; p < ann_network.branch_probs.size(); ++p) {
        for (size_t i = 0; i < ann_network.network.reticulation_nodes.size(); ++i) {
            Node *retNode = ann_network.network.reticulation_nodes[i];
            double firstParentProb = netrax::getReticulationFirstParentProb(retNode);
            double secondParentProb = netrax::getReticulationSecondParentProb(retNode);
            size_t firstParentEdgeIndex = netrax::getReticulationFirstParentPmatrixIndex(retNode);
            size_t secondParentEdgeIndex = netrax::getReticulationSecondParentPmatrixIndex(retNode);
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

}
