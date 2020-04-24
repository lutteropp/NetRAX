/*
 * AnnotatedNetwork.hpp
 *
 *  Created on: Feb 25, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <vector>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}
#include <raxml-ng/TreeInfo.hpp>
#include "Network.hpp"
#include "NetworkFunctions.hpp"
#include "BiconnectedComponents.hpp"
#include "../NetraxOptions.hpp"

namespace netrax {

struct AnnotatedNetwork {
    Network network; // The network topology itself
    TreeInfo* raxml_treeinfo;
    pllmod_treeinfo_t* fake_treeinfo = nullptr;
    NetraxOptions options;
    BlobInformation blobInfo; // mapping of edges to blobs, megablob roots, mapping of megablob roots to set of reticulation nodes within the megablob
    std::vector<Node*> topoTrav; // traversal in topological order
    std::vector<std::vector<double> > branch_probs; // for each partition, the branch length probs
};

}
