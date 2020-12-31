#include "Network.hpp"

namespace netrax {
    size_t Network::num_tips() const {
        return tipCount;
    }

    size_t Network::num_inner() const {
        return nodeCount - tipCount;
    }

    size_t Network::num_branches() const {
        return branchCount;
    }

    size_t Network::num_reticulations() const {
        return reticulation_nodes.size();
    }

    size_t Network::num_nodes() const {
        return nodeCount;
    }

    Node* Network::getNodeByLabel(const std::string &label) {
        Node *result = nullptr;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].getLabel() == label) {
                result = &nodes[i];
                break;
            }
        }
        assert(result);
        return result;
    }
    
    Network cloneNetwork(const Network& other) {
        Network network;
        network.nodeCount = other.nodeCount;
        network.branchCount = other.branchCount;
        network.tipCount = other.tipCount;

        network.nodes.resize(other.nodes.size());
        network.edges.resize(other.edges.size());

        // Clone the nodes and edges, first without any links
        for (size_t i = 0; i < other.nodes.size(); ++i) {
            if (other.nodes[i].type == NodeType::BASIC_NODE) {
                network.nodes[i].initBasic(other.nodes[i].clv_index, other.nodes[i].scaler_index, other.nodes[i].label);
            } else { // reticulation node
                ReticulationData* other_ret_data = other.nodes[i].getReticulationData().get();
                ReticulationData ret_data;
                ret_data.init(other_ret_data->reticulation_index, other_ret_data->label, other_ret_data->active_parent_toggle, nullptr, nullptr, nullptr);
                network.nodes[i].initReticulation(other.nodes[i].clv_index, other.nodes[i].scaler_index, other.nodes[i].label, ret_data);
            }
        }
        for (size_t i = 0; i < other.edges.size(); ++i) {
            network.edges[i].init(other.edges[i].pmatrix_index, nullptr, nullptr, other.edges[i].length, other.edges[i].prob);
        }

        // Fill the by-index references
        network.nodes_by_index.resize(other.nodes_by_index.size());
        network.edges_by_index.resize(other.edges_by_index.size());
        for (size_t i = 0; i < network.nodes.size(); ++i) {
            network.nodes_by_index[network.nodes[i].clv_index] = &network.nodes[i];
        }
        for (size_t i = 0; i < network.edges.size(); ++i) {
            network.edges_by_index[network.edges[i].pmatrix_index] = &network.edges[i];
        }
        network.reticulation_nodes.reserve(other.reticulation_nodes.size());
        for (size_t i = 0; i < other.reticulation_nodes.size(); ++i) {
            network.reticulation_nodes.emplace_back(network.nodes_by_index[other.reticulation_nodes[i]->clv_index]);
        }

        // Create the links
        for (size_t i = 0; i < other.nodes.size(); ++i) {
            for (size_t j = 0; j < other.nodes[i].links.size(); ++j) {
                Link link;
                link.init(other.nodes[i].links[j].node_clv_index, other.nodes[i].links[j].edge_pmatrix_index, nullptr, nullptr, other.nodes[i].links[j].direction);
                network.nodes[i].addLink(link);
            }
        }

        // Set the links for the edges
        for (size_t i = 0; i < network.nodes.size(); ++i) {
            for (size_t j = 0; j < network.nodes[i].links.size(); ++j) {
                if (network.nodes[i].links[j].direction == Direction::OUTGOING) {
                    network.edges_by_index[network.nodes[i].links[j].edge_pmatrix_index]->link1 = &network.nodes[i].links[j];
                } else {
                    network.edges_by_index[network.nodes[i].links[j].edge_pmatrix_index]->link2 = &network.nodes[i].links[j];
                }
            }
        }

        // Set the outer links
        for (size_t i = 0; i < network.edges.size(); ++i) {
            assert(network.edges[i].link1);
            assert(network.edges[i].link2);
            network.edges[i].link1->outer = network.edges[i].link2;
            network.edges[i].link2->outer = network.edges[i].link1;
        }

        // Set the links for the reticulation datas
        for (size_t i = 0; i < other.reticulation_nodes.size(); ++i) {
            ReticulationData* other_ret_data = other.nodes[i].getReticulationData().get();
            ReticulationData* my_ret_data = network.nodes[i].getReticulationData().get();

            size_t first_parent_edge_index = other_ret_data->link_to_first_parent->edge_pmatrix_index;
            size_t second_parent_edge_index = other_ret_data->link_to_second_parent->edge_pmatrix_index;
            size_t child_edge_index = other_ret_data->link_to_child->edge_pmatrix_index;

            my_ret_data->link_to_first_parent = network.edges_by_index[first_parent_edge_index]->link2; // because it is an incoming link
            my_ret_data->link_to_second_parent = network.edges_by_index[second_parent_edge_index]->link2; // because it is an incoming link
            my_ret_data->link_to_child = network.edges_by_index[child_edge_index]->link1; // because it is an outgoing link
        }

        // Set the root
        assert(other.root);
        network.root = network.nodes_by_index[other.root->clv_index];

        return network;
    }
}
