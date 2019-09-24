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
#include <iostream>

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
					unode->label ? unode->label : "");
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

				network.inner_nodes.push_back(&network.nodes[unode->clv_index]);
			} else { // leaf node, thus: only a single link
				network.links[unode->node_index].init(unode->node_index, &network.nodes[unode->clv_index],
						&network.edges[unode->pmatrix_index], nullptr, &network.links[unode->back->node_index], Direction::UNDEFINED);
				network.edges[unode->pmatrix_index].init(unode->pmatrix_index, &network.links[unode->node_index],
						&network.links[unode->back->node_index], unode->length);

				network.tip_nodes.push_back(&network.nodes[unode->clv_index]);
			}
		} else { // reticulation node
			ReticulationData retData;

			unetwork_node_t* first_parent_unode;
			unetwork_node_t* second_parent_unode;
			unetwork_node_t* child_unode;
			if (unode->incoming) {
				first_parent_unode = unode;
				if (unode->next->incoming) {
					second_parent_unode = unode->next;
					child_unode = unode->next->next;
				} else {
					child_unode = unode->next;
					second_parent_unode = unode->next->next;
				}
			} else {
				child_unode = unode;
				first_parent_unode = unode->next;
				second_parent_unode = unode->next->next;
			}
			assert(first_parent_unode->incoming);
			assert(second_parent_unode->incoming);
			assert(!child_unode->incoming);

			retData.init(first_parent_unode->reticulation_index, first_parent_unode->reticulation_name ? first_parent_unode->reticulation_name : "", 0,
					&network.links[first_parent_unode->node_index], &network.links[second_parent_unode->node_index],
					&network.links[child_unode->node_index], first_parent_unode->prob);
			network.nodes[first_parent_unode->clv_index].initReticulation(first_parent_unode->clv_index, first_parent_unode->scaler_index, &network.links[first_parent_unode->node_index],
					first_parent_unode->label ? first_parent_unode->label : "", retData);

			network.links[first_parent_unode->node_index].init(first_parent_unode->node_index, &network.nodes[first_parent_unode->clv_index], &network.edges[first_parent_unode->pmatrix_index],
					&network.links[second_parent_unode->node_index], &network.links[first_parent_unode->back->node_index], Direction::INCOMING);
			network.links[second_parent_unode->node_index].init(second_parent_unode->node_index, &network.nodes[second_parent_unode->clv_index],
					&network.edges[second_parent_unode->pmatrix_index], &network.links[child_unode->node_index],
					&network.links[second_parent_unode->back->node_index], Direction::INCOMING);
			network.links[child_unode->node_index].init(child_unode->node_index, &network.nodes[child_unode->clv_index],
					&network.edges[child_unode->pmatrix_index], &network.links[child_unode->node_index],
					&network.links[child_unode->back->node_index], Direction::OUTGOING);

			network.edges[first_parent_unode->pmatrix_index].init(first_parent_unode->pmatrix_index, &network.links[first_parent_unode->node_index],
					&network.links[first_parent_unode->back->node_index], first_parent_unode->length);
			network.edges[second_parent_unode->pmatrix_index].init(second_parent_unode->pmatrix_index, &network.links[second_parent_unode->node_index],
					&network.links[second_parent_unode->back->node_index], second_parent_unode->length);
			network.edges[child_unode->pmatrix_index].init(child_unode->pmatrix_index,
					&network.links[child_unode->node_index], &network.links[child_unode->back->node_index],
					child_unode->length);

			network.reticulation_nodes[first_parent_unode->reticulation_index] = &network.nodes[first_parent_unode->clv_index];
			network.inner_nodes.push_back(&network.nodes[first_parent_unode->clv_index]);
		}
	}
	network.root = &network.nodes[unetwork.vroot->clv_index];

	assert(network.tip_nodes.size() == unetwork.tip_count);
	assert(network.inner_nodes.size() == unetwork.inner_tree_count + unetwork.reticulation_count);

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
