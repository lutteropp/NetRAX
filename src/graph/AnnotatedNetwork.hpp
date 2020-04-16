/*
 * AnnotatedNetwork.hpp
 *
 *  Created on: Feb 25, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Network.hpp"
#include "NetworkFunctions.hpp"
#include "BiconnectedComponents.hpp"

namespace netrax {

struct AnnotatedNetwork {
    Network network; // The network topology itself
    std::vector<Node*> parent; // Pointers to the parent nodes of all network nodes
    BlobInformation blobInfo; // mapping of edges to blobs, megablob roots, mapping of megablob roots to set of reticulation nodes within the megablob
    std::vector<Node*> topoTrav; // traversal in topological order
};

}
