/*
 * NetworkConverter.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "NetworkIO.hpp"
#include "../Network.hpp"

#include <array>
#include <stdexcept>

namespace netrax {

// NetTree: Item[;]
// Item: (Item, Item) OR Node
// Node: TreeNode OR ReticulationNode

// TreeNode: [label][:branch_length] or [label] or [label][:branch_length][:support] or [] or [:branch_length] or [:branch_length][:support]
// ReticulationNode: [label]#[type]i[:branch_length] or [label][:branch_length][:support][:inheritance_probability] or #[type]i[:branch_length] or #[type]i[:branch_length][:support] or #[type]i[:branch_length][:support][:inheritance_probability] or #[type]i


Network convertNetwork(const unetwork_t& unetwork) {
	Network network;
	network.nodes.resize(unetwork.inner_tree_count + unetwork.tip_count + unetwork.reticulation_count);
	network.edges.resize(unetwork.edge_count);
	network.links.resize(unetwork.tip_count + unetwork.reticulation_count * 3 + unetwork.inner_tree_count * 3);
	network.reticulation_nodes.resize(unetwork.reticulation_count);

	for (size_t i = 0; i < network.nodes.size(); ++i) {
		unetwork_node_t* unode = unetwork.nodes[i];

		if (unode->reticulation_index == -1) { // basic node
			network.nodes[unode->clv_index].initBasic(unode->clv_index, unode->scaler_index, &network.links[unode->node_index],
					unode->label);
			if (unode->next) { // not a leaf node, thus: 3 links in total
				network.links[unode->node_index].init(unode->node_index, &network.nodes[unode->clv_index],
						&network.edges[unode->pmatrix_index], &network.links[unode->next->node_index],
						&network.links[unode->back->node_index], Direction::UNDEFINED);
				network.links[unode->next->node_index].init(unode->next->node_index, &network.nodes[unode->clv_index],
						&network.edges[unode->next->pmatrix_index], &network.links[unode->next->next->node_index],
						&network.links[unode->next->back->node_index], Direction::UNDEFINED);
				network.links[unode->next->next->node_index].init(unode->next->next->node_index, &network.nodes[unode->clv_index],
						&network.edges[unode->next->next->pmatrix_index], &network.links[unode->node_index],
						&network.links[unode->next->next->back->node_index], Direction::UNDEFINED);

				network.edges[unode->pmatrix_index].init(unode->pmatrix_index, &network.links[unode->node_index],
						&network.links[unode->back->node_index], unode->length);
				network.edges[unode->next->pmatrix_index].init(unode->next->pmatrix_index, &network.links[unode->next->node_index],
						&network.links[unode->next->back->node_index], unode->next->length);
				network.edges[unode->next->next->pmatrix_index].init(unode->next->next->pmatrix_index,
						&network.links[unode->next->next->node_index], &network.links[unode->next->next->back->node_index],
						unode->next->next->length);

			} else { // leaf node, thus: only a single link
				network.links[unode->node_index].init(unode->node_index, &network.nodes[unode->clv_index],
						&network.edges[unode->pmatrix_index], nullptr, &network.links[unode->back->node_index], Direction::UNDEFINED);
				network.edges[unode->pmatrix_index].init(unode->pmatrix_index, &network.links[unode->node_index],
						&network.links[unode->back->node_index], unode->length);
			}
		} else { // reticulation node
			ReticulationData retData;
			assert(unode->incoming); // link to first parent
			assert(unode->next->incoming); // link to second parent
			assert(!unode->next->next->incoming); // link to child
			assert(unode->next->next->next == unode);

			retData.init(unode->reticulation_index, unode->reticulation_name, 0, &network.links[unode->node_index],
					&network.links[unode->next->node_index], &network.links[unode->next->next->node_index], unode->prob);
			network.nodes[unode->clv_index].initReticulation(unode->clv_index, unode->scaler_index, &network.links[unode->node_index],
					unode->label, retData);

			network.links[unode->node_index].init(unode->node_index, &network.nodes[unode->clv_index], &network.edges[unode->pmatrix_index],
					&network.links[unode->next->node_index], &network.links[unode->back->node_index], Direction::INCOMING);
			network.links[unode->next->node_index].init(unode->next->node_index, &network.nodes[unode->clv_index],
					&network.edges[unode->next->pmatrix_index], &network.links[unode->next->next->node_index],
					&network.links[unode->next->back->node_index], Direction::INCOMING);
			network.links[unode->next->next->node_index].init(unode->next->next->node_index, &network.nodes[unode->clv_index],
					&network.edges[unode->next->next->pmatrix_index], &network.links[unode->node_index],
					&network.links[unode->next->next->back->node_index], Direction::OUTGOING);

			network.edges[unode->pmatrix_index].init(unode->pmatrix_index, &network.links[unode->node_index],
					&network.links[unode->back->node_index], unode->length);
			network.edges[unode->next->pmatrix_index].init(unode->next->pmatrix_index, &network.links[unode->next->node_index],
					&network.links[unode->next->back->node_index], unode->next->length);
			network.edges[unode->next->next->pmatrix_index].init(unode->next->next->pmatrix_index,
					&network.links[unode->next->next->node_index], &network.links[unode->next->next->back->node_index],
					unode->next->next->length);

			network.reticulation_nodes[unode->reticulation_index] = &network.nodes[unode->clv_index];
		}
	}
	network.root = &network.nodes[unetwork.vroot->clv_index];
	network.tip_count = unetwork.tip_count;

	return network;
}

Network readNetworkFromString(const std::string& newick) {
	unetwork_t * unetwork = unetwork_parse_newick_string(newick.c_str());
	return convertNetwork(*unetwork);
}
Network readNetworkFromFile(const std::string& filename) {
	unetwork_t * unetwork = unetwork_parse_newick(filename.c_str());
	return convertNetwork(*unetwork);
}
std::string toExtendedNewick(const Network& network) {
	throw std::runtime_error("This function has not been implemented yet");
}

}
