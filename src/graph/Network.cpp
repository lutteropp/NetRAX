#include "Network.hpp"
#include <iostream>

namespace netrax
{
    size_t Network::num_tips() const
    {
        return tipCount;
    }

    size_t Network::num_inner() const
    {
        return nodeCount - tipCount;
    }

    size_t Network::num_branches() const
    {
        return branchCount;
    }

    size_t Network::num_reticulations() const
    {
        return reticulation_nodes.size();
    }

    size_t Network::num_nodes() const
    {
        return nodeCount;
    }

    Node *Network::getNodeByLabel(const std::string &label)
    {
        Node *result = nullptr;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            if (nodes[i].getLabel() == label)
            {
                result = &nodes[i];
                break;
            }
        }
        assert(result);
        return result;
    }

    void cloneNodesAndEdges(Network &network, const Network &other)
    {
        for (size_t i = 0; i < other.nodes.size(); ++i)
        {
            if (other.nodes[i].type == NodeType::BASIC_NODE)
            {
                network.nodes[i].initBasic(other.nodes[i].clv_index, other.nodes[i].scaler_index, other.nodes[i].label);
            }
            else
            { // reticulation node
                ReticulationData *other_ret_data = other.nodes[i].getReticulationData().get();
                ReticulationData ret_data;
                ret_data.init(other_ret_data->reticulation_index, other_ret_data->label, other_ret_data->active_parent_toggle, nullptr, nullptr, nullptr);
                network.nodes[i].initReticulation(other.nodes[i].clv_index, other.nodes[i].scaler_index, other.nodes[i].label, ret_data);
            }
        }
        for (size_t i = 0; i < other.edges.size(); ++i)
        {
            network.edges[i].init(other.edges[i].pmatrix_index, nullptr, nullptr, other.edges[i].length, other.edges[i].prob);
        }
    }

    void fillByIndexReferences(Network &network, const Network &other)
    {
        network.nodes_by_index.resize(other.nodes_by_index.size());
        network.edges_by_index.resize(other.edges_by_index.size());
        for (size_t i = 0; i < network.nodes.size(); ++i)
        {
            if (network.nodes[i].clv_index == std::numeric_limits<size_t>::max())
            {
                continue;
            }
            network.nodes_by_index[network.nodes[i].clv_index] = &network.nodes[i];
        }
        for (size_t i = 0; i < network.edges.size(); ++i)
        {
            if (network.edges[i].pmatrix_index == std::numeric_limits<size_t>::max())
            {
                continue;
            }
            network.edges_by_index[network.edges[i].pmatrix_index] = &network.edges[i];
        }
        network.reticulation_nodes.reserve(other.reticulation_nodes.size());
        for (size_t i = 0; i < other.reticulation_nodes.size(); ++i)
        {
            network.reticulation_nodes.emplace_back(network.nodes_by_index[other.reticulation_nodes[i]->clv_index]);
        }
    }

    void createTheLinks(Network &network, const Network &other)
    {
        assert(network.num_nodes() == other.num_nodes());
        for (size_t i = 0; i < other.num_nodes(); ++i)
        {
            for (size_t j = 0; j < other.nodes_by_index[i]->links.size(); ++j)
            {
                Link link;
                link.init(other.nodes_by_index[i]->links[j].node_clv_index, other.nodes_by_index[i]->links[j].edge_pmatrix_index, nullptr, nullptr, other.nodes_by_index[i]->links[j].direction);
                network.nodes_by_index[i]->addLink(link);
            }
        }
        for (size_t i = 0; i < other.num_tips(); ++i) {
            assert(other.nodes_by_index[i]->links.size() == 1);
            assert(network.nodes_by_index[i]->links.size() == 1);
        }
    }

    void setLinksForEdges(Network &network)
    {
        for (size_t i = 0; i < network.nodes.size(); ++i)
        {
            for (size_t j = 0; j < network.nodes[i].links.size(); ++j)
            {
                size_t pmatrix_index = network.nodes[i].links[j].edge_pmatrix_index;
                if (network.nodes[i].links[j].direction == Direction::OUTGOING)
                {
                    assert(!network.edges_by_index[pmatrix_index]->link1);
                    network.edges_by_index[pmatrix_index]->link1 = &network.nodes[i].links[j];
                }
                else
                {
                    assert(!network.edges_by_index[pmatrix_index]->link2);
                    network.edges_by_index[pmatrix_index]->link2 = &network.nodes[i].links[j];
                }
            }
        }
    }

    void setOuterLinks(Network &network)
    {
        for (size_t i = 0; i < network.edges.size(); ++i)
        {
            if (network.edges[i].pmatrix_index == std::numeric_limits<size_t>::max())
            {
                continue;
            }
            assert(network.edges_by_index[network.edges[i].pmatrix_index] == &network.edges[i]);
            assert(network.edges[i].link1);
            assert(network.edges[i].link2);
            network.edges[i].link1->outer = network.edges[i].link2;
            network.edges[i].link2->outer = network.edges[i].link1;
        }
    }

    void setReticulationLinks(Network &network, const Network &other)
    {
        for (size_t i = 0; i < other.reticulation_nodes.size(); ++i)
        {
            ReticulationData *other_ret_data = other.reticulation_nodes[i]->getReticulationData().get();
            ReticulationData *my_ret_data = network.reticulation_nodes[i]->getReticulationData().get();

            size_t first_parent_edge_index = other_ret_data->link_to_first_parent->edge_pmatrix_index;
            size_t second_parent_edge_index = other_ret_data->link_to_second_parent->edge_pmatrix_index;
            size_t child_edge_index = other_ret_data->link_to_child->edge_pmatrix_index;

            my_ret_data->link_to_first_parent = network.edges_by_index[first_parent_edge_index]->link2;   // because it is an incoming link
            my_ret_data->link_to_second_parent = network.edges_by_index[second_parent_edge_index]->link2; // because it is an incoming link
            my_ret_data->link_to_child = network.edges_by_index[child_edge_index]->link1;                 // because it is an outgoing link
        }
    }

    Network::Network(const Network &other)
    {
        nodeCount = other.nodeCount;
        branchCount = other.branchCount;
        tipCount = other.tipCount;

        nodes.resize(other.nodes.size());
        edges.resize(other.edges.size());

        // Clone the nodes and edges, first without any links
        cloneNodesAndEdges(*this, other);

        // Fill the by-index references
        fillByIndexReferences(*this, other);

        // Create the links
        createTheLinks(*this, other);

        // Set the links for the edges
        setLinksForEdges(*this);

        // Set the outer links
        setOuterLinks(*this);

        // Set the links for the reticulation datas
        setReticulationLinks(*this, other);

        // Set the root
        assert(other.root);
        root = nodes_by_index[other.root->clv_index];
    }
} // namespace netrax
