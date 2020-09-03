#pragma once

#include "Network.hpp"

#include <vector>
#include <memory>

namespace netrax {

/**
 * The blob information data structure. 
 * 
 * A blob is a biconnected component (see https://en.wikipedia.org/wiki/Biconnected_component) in the underlying undirected graph of the phylogenetic network.
 * 
 * We use blobs in optimizing the loglikelihood computation for a network, because we can compute the loglikelihood within each blob separately. 
 * We form a component graph with the blobs as its nodes and have an edge between two blobs in the component graph, if and only if the two blobs share a node with each other.
 * The component graph we obtain is always a tree, as non tree-like structures are hidden away within the blobs.
 * 
 * On this component graph, we can use the Felsenstein pruning algorithm for loglikelihood computation on a tree.
 * 
 * For each edge in the network, we assign it a edge_blob_id, stating to which blob the edge belongs. The blob_size of a blob is the number of edges included in the blob.
 * A node can belong to multiple blobs. However, we set the node_blob_id as TODO: How is the node_blob_id assigned?
 * 
 * A reticulation node can never be a megablob root, as it is hidden away in a blob. A non-reticulation node is a megablob root, if and only if:
 *  1.) It is the root node, or
 *  2.) Its parent has another node_blob_id, belonging to a blob with blob_size > 1
 * 
 * It is sufficient to compute the loglikelihood of the subnetworks rooted at the megablob roots. 
 * To get the final tree we apply Felsenstein pruning algorithm to, we do what would be a post-order traversal in a tree. 
 * For our phylogenetic network which is a rooted single-source DAG, we do a reversed topological sort to reach an equivalent bottom-up behavior.
 * 
 * Whenever we encounter a megablob root in our traversal, we compute the CLV (conditional loglikelihood vector, TODO: explain) for the subnetwork rooted at this megablob root, then discard/collapse all nodes below the megablob root.
 * We never actually build any graph structures mentioned here, but instead continue to operate on the phylogenetic network itself. This explanation is just for easier understanding of the general underlying approach.
 */
struct BlobInformation {
    std::vector<unsigned int> edge_blob_id;
    std::vector<unsigned int> node_blob_id; // TODO: Maybe remove this again, as it is only used in the debug output
    std::vector<std::vector<Node*> > reticulation_nodes_per_megablob;
    std::vector<Node*> megablob_roots;
};
BlobInformation partitionNetworkIntoBlobs(Network &network, const std::vector<Node*> &travbuffer);
}
