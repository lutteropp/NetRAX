#include "Helper.hpp"

namespace netrax {

Node* getTargetNode(const Network &network, const Link *link) {
    assert(link);
    if (network.edges_by_index[link->edge_pmatrix_index]->link1 == link) {
        return network.nodes_by_index[network.edges_by_index[link->edge_pmatrix_index]->link2->node_clv_index];
    } else {
        assert(network.edges_by_index[link->edge_pmatrix_index]->link2 == link);
        return network.nodes_by_index[network.edges_by_index[link->edge_pmatrix_index]->link1->node_clv_index];
    }
}

Link* make_link(Node *node, Edge *edge, Direction dir) {
    Link link;
    link.init(node ? node->clv_index : 0, edge ? edge->pmatrix_index : 0, nullptr, nullptr, dir);
    return node->addLink(link);
}

std::vector<Link*> getLinksToClvIndex(Network &network, Node *node, size_t target_index) {
    assert(node);
    std::vector<Link*> res;
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(network, &(node->links[i]))->clv_index == target_index) {
            res.emplace_back(&(node->links[i]));
        }
    }
    return res;
}

Link* getLinkToNode(Network &network, Node *node, Node *target) {
    assert(node);
    for (size_t i = 0; i < node->links.size(); ++i) {
        if (getTargetNode(network, &(node->links[i])) == target) {
            return &(node->links[i]);
        }
    }
    throw std::runtime_error("the node is not a neighbor");
}

Link* getLinkToNode(Network &network, size_t from_clv_index, size_t to_clv_index) {
    return getLinkToNode(network, network.nodes_by_index[from_clv_index],
            network.nodes_by_index[to_clv_index]);
}

}