/*
 * BiconnectedComponents.hpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "Network.hpp"

#include <vector>
#include <memory>

namespace netrax {
// A blob is a biconnected component in the underlying undirected graph of the phylogenetic network.
struct BlobInformation {
    std::vector<unsigned int> edge_blob_id;
    std::vector<unsigned int> node_blob_id; // TODO: Maybe remove this again, as it is only used in the debug output
    std::vector<std::vector<Node*> > reticulation_nodes_per_megablob;
    std::vector<Node*> megablob_roots;
};
BlobInformation partitionNetworkIntoBlobs(Network &network, const std::vector<Node*> &travbuffer);
}
