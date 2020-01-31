/*
 * BiconnectedComponents.hpp
 *
 *  Created on: Jan 17, 2020
 *      Author: Sarah Lutteropp
 */

#include "Network.hpp"

#include <vector>
#include <memory>

namespace netrax {
	// A blob is a biconnected component in the underlying undirected graph of the phylogenetic network.
	struct BlobInformation {
		std::vector<unsigned int> edge_blob_id;
		std::vector<unsigned int> blob_size;
		std::vector<std::vector<const Node*> > reticulation_nodes_per_blob;
		std::vector<const Node*> megablob_roots;
	};
	BlobInformation partitionNetworkIntoBlobs(const Network& network);
}
