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
	};
	BlobInformation partitionNetworkEdgesIntoBlobs(const Network& network);

	struct BlobNode;
	struct BlobEdge {
		double length;
		std::unique_ptr<BlobNode> child;
	};

	struct BlobNode {
		unsigned int id;
		Network blobNetwork;
		std::vector<BlobEdge> children;
	};

	struct BlobTree {
		std::unique_ptr<BlobNode> root;
	};
}
