#include "Helper.hpp"
#include "NetworkFunctions.hpp"
namespace netrax {

Edge* getEdgeTo(Network &network, const Node *node, const Node *target) {
    assert(node);
    assert(target);
    for (const auto &link : node->links) {
        if (link.outer->node_clv_index == target->clv_index) {
            return network.edges_by_index[link.edge_pmatrix_index];
        }
    }
    throw std::runtime_error("The given target node is not a neighbor of this node");
}

Edge* getEdgeTo(Network &network, size_t from_clv_index, size_t to_clv_index) {
    return getEdgeTo(network, network.nodes_by_index[from_clv_index],
            network.nodes_by_index[to_clv_index]);
}

std::vector<Edge*> getAdjacentEdges(Network &network, const Node *node) {
    std::vector<Edge*> res;
    std::vector<Node*> neighs = getNeighbors(network, node);
    for (size_t i = 0; i < neighs.size(); ++i) {
        res.emplace_back(getEdgeTo(network, node, neighs[i]));
    }
    assert(res.size() == 1 || res.size() == 3 || (node == network.root && res.size() == 2));
    return res;
}

std::vector<Edge*> getAdjacentEdges(Network &network, const Edge *edge) {
    std::vector<Edge*> res;

    Node *node1 = network.nodes_by_index[edge->link1->node_clv_index];
    Node *node2 = network.nodes_by_index[edge->link2->node_clv_index];

    for (size_t i = 0; i < node1->links.size(); ++i) {
        if (node1->links[i].edge_pmatrix_index != edge->pmatrix_index) {
            Edge *neigh = network.edges_by_index[node1->links[i].edge_pmatrix_index];
            if (std::find(res.begin(), res.end(), neigh) == res.end()) {
                res.emplace_back(neigh);
            }
        }
    }

    for (size_t i = 0; i < node2->links.size(); ++i) {
        if (node2->links[i].edge_pmatrix_index != edge->pmatrix_index) {
            Edge *neigh = network.edges_by_index[node2->links[i].edge_pmatrix_index];
            if (std::find(res.begin(), res.end(), neigh) == res.end()) {
                res.emplace_back(neigh);
            }
        }
    }
    assert(
            res.size() == 2 || res.size() == 4 || (node1 == network.root && res.size() == 3)
                    || (node1 == network.root && node2->isTip() && res.size() == 1));
    return res;
}

Node* getSource(Network &network, const Edge *edge) {
    assert(edge);
    assert(edge->link1);
    assert(edge->link2);
    assert(edge->link1->direction == Direction::OUTGOING);
    return network.nodes_by_index[edge->link1->node_clv_index];
}

Node* getTarget(Network &network, const Edge *edge) {
    assert(edge);
    assert(edge->link1);
    assert(edge->link2);
    assert(edge->link2->direction == Direction::INCOMING);
    return network.nodes_by_index[edge->link2->node_clv_index];
}

bool isOutgoing(Network &network, const Node *from, const Node *to) {
    assert(!getLinksToClvIndex(network, from, to->clv_index).empty());
    auto children = getChildren(network, from);
    return (std::find(children.begin(), children.end(), to) != children.end());
}

bool isActiveBranch(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, unsigned int pmatrix_index) {
    const Node* edge_source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    const Node* edge_target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);

    ReticulationConfigSet restrictions = getRestrictionsToTakeNeighbor(ann_network, edge_source, edge_target);
    return reticulationConfigsCompatible(restrictions, reticulationChoices);
}

bool isActiveAliveBranch(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, const std::vector<bool>& dead_nodes, unsigned int pmatrix_index) {
    const Node* edge_source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    const Node* edge_target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);

    ReticulationConfigSet restrictions = getRestrictionsToTakeNeighbor(ann_network, edge_source, edge_target);
    return reticulationConfigsCompatible(restrictions, reticulationChoices) && !dead_nodes[edge_source->clv_index] && !dead_nodes[edge_target->clv_index];
}

bool isActiveAliveBranch(AnnotatedNetwork& ann_network, const ReticulationConfigSet& reticulationChoices, unsigned int pmatrix_index) {
    setReticulationParents(ann_network.network, reticulationChoices.configs[0]);
    std::vector<bool> dead_nodes = collect_dead_nodes(ann_network.network, ann_network.network.root->clv_index);
    return isActiveAliveBranch(ann_network, reticulationChoices, dead_nodes, pmatrix_index);
}

}