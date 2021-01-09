/*
 * Moves.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#include "Moves.hpp"

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "../graph/AnnotatedNetwork.hpp"
#include "../graph/BiconnectedComponents.hpp"
#include "../graph/Direction.hpp"
#include "../graph/Edge.hpp"
#include "../graph/Link.hpp"
#include "../graph/Network.hpp"
#include "../graph/NetworkFunctions.hpp"
#include "../graph/NetworkTopology.hpp"
#include "../graph/Node.hpp"
#include "../graph/NodeType.hpp"
#include "../graph/ReticulationData.hpp"
#include "../NetraxOptions.hpp"

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

std::vector<double> get_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index) {
    std::vector<double> res(ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1);
    for (size_t p = 0; p < res.size(); ++p) {
        res[p] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        assert(res[p] >= ann_network.options.brlen_min);
        assert(res[p] <= ann_network.options.brlen_max);
    }
    return res;
}

void set_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index, const std::vector<double> &lengths) {
    for (size_t p = 0; p < lengths.size(); ++p) {
        ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] = lengths[p];
        assert(lengths[p] >= ann_network.options.brlen_min);
        assert(lengths[p] <= ann_network.options.brlen_max);
    }
}

bool hasPath(Network &network, const Node *from, const Node *to, bool nonelementary = false) {
    std::vector<bool> visited(network.num_nodes(), false);
    std::queue<std::pair<const Node*, const Node*> > q;
    q.emplace(to, nullptr);
    while (!q.empty()) {
        const Node *node = q.front().first;
        const Node *child = q.front().second;
        if (node == from) {
            if (!nonelementary || child != to) {
                return true;
            }
        }
        q.pop();
        visited[node->clv_index] = true;
        for (const Node *neigh : getAllParents(network, node)) {
            if (!visited[neigh->clv_index] || (nonelementary && neigh == from)) {
                q.emplace(std::make_pair(neigh, node));
            }
        }
    }
    return false;
}

/*
 * we need to choose s and t in a way that there are elementary connections {u,s} and {v,t},
 * but there are no elementary connections {u,t} and {v,s}
 */
std::vector<std::pair<Node*, Node*> > getSTChoices(Network &network, const Edge *edge) {
    std::vector<std::pair<Node*, Node*> > res;
    Node *u = getSource(network, edge);
    Node *v = getTarget(network, edge);

    auto uNeighbors = getNeighbors(network, u);
    auto vNeighbors = getNeighbors(network, v);

    for (const auto &s : uNeighbors) {
        if (s == v)
            continue;
        for (const auto &t : vNeighbors) {
            if (t == u)
                continue;

            if (std::find(uNeighbors.begin(), uNeighbors.end(), t) == uNeighbors.end()
                    && std::find(vNeighbors.begin(), vNeighbors.end(), s) == vNeighbors.end()) {
                res.emplace_back(std::make_pair(s, t));
            }
        }
    }
    return res;
}

RNNIMove buildRNNIMove(size_t u_clv_index, size_t v_clv_index, size_t s_clv_index,
        size_t t_clv_index, RNNIMoveType type) {
    RNNIMove move = RNNIMove();
    move.u_clv_index = u_clv_index;
    move.v_clv_index = v_clv_index;
    move.s_clv_index = s_clv_index;
    move.t_clv_index = t_clv_index;
    move.type = type;
    return move;
}

std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network, const Edge *edge) {
    Network &network = ann_network.network;
    std::vector<RNNIMove> res;
    Node *u = getSource(network, edge);
    Node *v = getTarget(network, edge);
    auto stChoices = getSTChoices(network, edge);
    for (const auto &st : stChoices) {
        Node *s = st.first;
        Node *t = st.second;

        // check for possible variant and add move from the paper if the move would not create a cycle
        if (isOutgoing(network, u, s) && isOutgoing(network, v, t)) {
            if (!hasPath(network, s, v)) {
                // add move 1
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::ONE));
                if (v->type == NodeType::RETICULATION_NODE && u != network.root) {
                    // add move 1*
                    res.emplace_back(
                            buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                    RNNIMoveType::ONE_STAR));
                }
            }
        } else if (isOutgoing(network, s, u) && isOutgoing(network, t, v)) {
            if (!hasPath(network, u, t)) {
                // add move 2
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::TWO));
                if (u->type != NodeType::RETICULATION_NODE) {
                    // add move 2*
                    res.emplace_back(
                            buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                    RNNIMoveType::TWO_STAR));
                }
            }
        } else if (isOutgoing(network, s, u) && isOutgoing(network, v, t)) {
            if (u->type == NodeType::RETICULATION_NODE && v->type != NodeType::RETICULATION_NODE) {
                // add move 3
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::THREE));
            }
            if (!hasPath(network, u, v, true)) {
                // add move 3*
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::THREE_STAR));
            }
        } else if (isOutgoing(network, u, s) && isOutgoing(network, t, v)) {
            if (u != network.root && !hasPath(network, s, t)) {
                // add move 4
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::FOUR));
            }
        }
    }
    return res;
}

void exchangeEdges(Network &network, Node *u, Node *v, Node *s, Node *t) {
    // The edge between {u,s} will now be between {u, t} and the edge between {v,t} will now be between {v,s}. The edge directions stay the same.
    Link *from_u_link = getLinkToNode(network, u, s);
    Link *from_s_link = getLinkToNode(network, s, u);
    Link *from_v_link = getLinkToNode(network, v, t);
    Link *from_t_link = getLinkToNode(network, t, v);
    Edge *u_s_edge = getEdgeTo(network, u, s);
    Edge *v_t_edge = getEdgeTo(network, v, t);

    from_u_link->outer = from_t_link;
    from_t_link->outer = from_u_link;
    from_v_link->outer = from_s_link;
    from_s_link->outer = from_v_link;

    assert(from_u_link->node_clv_index != from_u_link->outer->node_clv_index);
    assert(from_t_link->node_clv_index != from_t_link->outer->node_clv_index);
    assert(from_v_link->node_clv_index != from_v_link->outer->node_clv_index);
    assert(from_s_link->node_clv_index != from_s_link->outer->node_clv_index);

    // u_s_edge now becomes u_t edge
    Edge *u_t_edge = u_s_edge;
    u_t_edge->link1 = from_u_link;
    u_t_edge->link2 = from_t_link;
    from_u_link->edge_pmatrix_index = u_t_edge->pmatrix_index;
    from_t_link->edge_pmatrix_index = u_t_edge->pmatrix_index;
    if (u_t_edge->link1->direction == Direction::INCOMING) {
        std::swap(u_t_edge->link1, u_t_edge->link2);
    }

    // v_t_edge now becomes v_s_edge
    Edge *v_s_edge = v_t_edge;
    v_s_edge->link1 = from_v_link;
    v_s_edge->link2 = from_s_link;
    from_v_link->edge_pmatrix_index = v_s_edge->pmatrix_index;
    from_s_link->edge_pmatrix_index = v_s_edge->pmatrix_index;
    if (v_s_edge->link1->direction == Direction::INCOMING) {
        std::swap(v_s_edge->link1, v_s_edge->link2);
    }
}

void switchReticulations(Network &network, Node *u, Node *v) {
    Node *old_ret_node;
    Node *new_ret_node;
    if (u->type == NodeType::RETICULATION_NODE) {
        assert(v->type != NodeType::RETICULATION_NODE);
        old_ret_node = u;
        new_ret_node = v;
    } else {
        assert(v->type == NodeType::RETICULATION_NODE);
        old_ret_node = v;
        new_ret_node = u;
    }
    size_t reticulationId = old_ret_node->getReticulationData()->reticulation_index;
    network.reticulation_nodes[reticulationId] = new_ret_node;

    std::string label = old_ret_node->reticulationData->label;
    bool active = old_ret_node->reticulationData->active_parent_toggle;
    old_ret_node->reticulationData.release();

    Link *link_to_first_parent = nullptr;
    Link *link_to_second_parent = nullptr;
    Link *link_to_child = nullptr;
    for (auto &link : new_ret_node->links) {
        if (link.direction == Direction::OUTGOING) {
            link_to_child = &link;
        } else {
            if (link_to_first_parent == nullptr) {
                link_to_first_parent = &link;
            } else {
                link_to_second_parent = &link;
            }
        }
    }
    assert(link_to_first_parent);
    assert(link_to_second_parent);
    assert(link_to_child);

    if (link_to_first_parent->edge_pmatrix_index > link_to_second_parent->edge_pmatrix_index) {
        std::swap(link_to_first_parent, link_to_second_parent);
    }

    ReticulationData retData;
    retData.init(reticulationId, label, active, link_to_first_parent, link_to_second_parent,
            link_to_child);
    new_ret_node->reticulationData = std::make_unique<ReticulationData>(retData);

    old_ret_node->type = NodeType::BASIC_NODE;
    new_ret_node->type = NodeType::RETICULATION_NODE;
}

void resetReticulationLinks(Node *node) {
    assert(node->type == NodeType::RETICULATION_NODE);
    auto retData = node->getReticulationData().get();
    retData->link_to_first_parent = nullptr;
    retData->link_to_second_parent = nullptr;
    retData->link_to_child = nullptr;
    for (Link &link : node->links) {
        if (link.direction == Direction::OUTGOING) {
            retData->link_to_child = &link;
        } else if (retData->link_to_first_parent == nullptr) {
            retData->link_to_first_parent = &link;
        } else {
            retData->link_to_second_parent = &link;
        }
    }
    assert(retData->link_to_first_parent);
    assert(retData->link_to_second_parent);
    assert(retData->link_to_child);
    if (retData->link_to_first_parent->edge_pmatrix_index
            > retData->link_to_second_parent->edge_pmatrix_index) {
        std::swap(retData->link_to_first_parent, retData->link_to_second_parent);
    }
}

void fixReticulationLinks(Node *u, Node *v, Node *s, Node *t) {
    if (u->type == NodeType::RETICULATION_NODE)
        resetReticulationLinks(u);
    if (v->type == NodeType::RETICULATION_NODE)
        resetReticulationLinks(v);
    if (s->type == NodeType::RETICULATION_NODE)
        resetReticulationLinks(s);
    if (t->type == NodeType::RETICULATION_NODE)
        resetReticulationLinks(t);
}

void addRepairCandidates(Network &network, std::unordered_set<Node*> &repair_candidates,
        Node *node) {
    repair_candidates.emplace(node);
    for (Node *neigh : getNeighbors(network, node)) {
        repair_candidates.emplace(neigh);
    }
}

void fixReticulations(Network &network, RNNIMove &move) {
    std::unordered_set<Node*> repair_candidates;
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.s_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.t_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.u_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.v_clv_index]);
    for (Node *node : repair_candidates) {
        if (node->type == NodeType::RETICULATION_NODE) {
            resetReticulationLinks(node);
        }
    }
}

void fixReticulations(Network &network, RSPRMove &move) {
    std::unordered_set<Node*> repair_candidates;
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.x_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.x_prime_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.y_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.y_prime_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.z_clv_index]);
    for (Node *node : repair_candidates) {
        if (node->type == NodeType::RETICULATION_NODE) {
            resetReticulationLinks(node);
        }
    }
}

void fixReticulations(Network &network, ArcRemovalMove &move) {
    std::unordered_set<Node*> repair_candidates;
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.a_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.b_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.c_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.d_clv_index]);
    for (Node *node : repair_candidates) {
        if (node->type == NodeType::RETICULATION_NODE) {
            resetReticulationLinks(node);
        }
    }
}

void fixReticulations(Network &network, ArcInsertionMove &move) {
    // change parent links from reticulation nodes such that link_to_first_parent points to the smaller pmatrix index
    std::unordered_set<Node*> repair_candidates;
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.a_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.b_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.c_clv_index]);
    addRepairCandidates(network, repair_candidates, network.nodes_by_index[move.d_clv_index]);
    addRepairCandidates(network, repair_candidates,
            network.nodes_by_index[move.wanted_u_clv_index]);
    addRepairCandidates(network, repair_candidates,
            network.nodes_by_index[move.wanted_v_clv_index]);
    for (Node *node : repair_candidates) {
        if (node->type == NodeType::RETICULATION_NODE) {
            resetReticulationLinks(node);
        }
    }
}

void changeEdgeDirection(Network &network, Node *u, Node *v) {
    Link *from_u_link = getLinkToNode(network, u, v);
    Link *from_v_link = getLinkToNode(network, v, u);
    if (from_u_link->direction == Direction::INCOMING) {
        assert(from_v_link->direction == Direction::OUTGOING);
        from_u_link->direction = Direction::OUTGOING;
        from_v_link->direction = Direction::INCOMING;
    } else {
        assert(from_v_link->direction == Direction::INCOMING);
        from_u_link->direction = Direction::INCOMING;
        from_v_link->direction = Direction::OUTGOING;
    }
}

void setLinkDirections(Network &network, Node *u, Node *v) {
    Link *from_u_link = getLinkToNode(network, u, v);
    Link *from_v_link = getLinkToNode(network, v, u);
    from_u_link->direction = Direction::OUTGOING;
    from_v_link->direction = Direction::INCOMING;
}

void updateLinkDirections(Network &network, RNNIMove &move) {
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    switch (move.type) {
    case RNNIMoveType::ONE:
        setLinkDirections(network, u, v);
        setLinkDirections(network, u, t);
        setLinkDirections(network, v, s);
        break;
    case RNNIMoveType::ONE_STAR:
        setLinkDirections(network, v, u);
        setLinkDirections(network, u, t);
        setLinkDirections(network, v, s);
        break;
    case RNNIMoveType::TWO:
        setLinkDirections(network, u, v);
        setLinkDirections(network, t, u);
        setLinkDirections(network, s, v);
        break;
    case RNNIMoveType::TWO_STAR:
        setLinkDirections(network, v, u);
        setLinkDirections(network, t, u);
        setLinkDirections(network, s, v);
        break;
    case RNNIMoveType::THREE:
        setLinkDirections(network, u, v);
        setLinkDirections(network, u, t);
        setLinkDirections(network, s, v);
        break;
    case RNNIMoveType::THREE_STAR:
        setLinkDirections(network, v, u);
        setLinkDirections(network, u, t);
        setLinkDirections(network, s, v);
        break;
    case RNNIMoveType::FOUR:
        setLinkDirections(network, u, v);
        setLinkDirections(network, t, u);
        setLinkDirections(network, v, s);
        break;
    }

    Edge *u_v_edge = getEdgeTo(network, u, v);
    if (u_v_edge->link1->direction == Direction::INCOMING) {
        std::swap(u_v_edge->link1, u_v_edge->link2);
    }
    Edge *u_t_edge = getEdgeTo(network, u, t);
    if (u_t_edge->link1->direction == Direction::INCOMING) {
        std::swap(u_t_edge->link1, u_t_edge->link2);
    }
    Edge *v_s_edge = getEdgeTo(network, v, s);
    if (v_s_edge->link1->direction == Direction::INCOMING) {
        std::swap(v_s_edge->link1, v_s_edge->link2);
    }
}

void updateLinkDirectionsReverse(Network &network, RNNIMove &move) {
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    switch (move.type) {
    case RNNIMoveType::ONE:
        setLinkDirections(network, u, s);
        setLinkDirections(network, u, v);
        setLinkDirections(network, v, t);
        break;
    case RNNIMoveType::ONE_STAR:
        setLinkDirections(network, u, s);
        setLinkDirections(network, u, v);
        setLinkDirections(network, v, t);
        break;
    case RNNIMoveType::TWO:
        setLinkDirections(network, s, u);
        setLinkDirections(network, u, v);
        setLinkDirections(network, t, v);
        break;
    case RNNIMoveType::TWO_STAR:
        setLinkDirections(network, s, u);
        setLinkDirections(network, u, v);
        setLinkDirections(network, t, v);
        break;
    case RNNIMoveType::THREE:
        setLinkDirections(network, s, u);
        setLinkDirections(network, u, v);
        setLinkDirections(network, v, t);
        break;
    case RNNIMoveType::THREE_STAR:
        setLinkDirections(network, s, u);
        setLinkDirections(network, u, v);
        setLinkDirections(network, v, t);
        break;
    case RNNIMoveType::FOUR:
        setLinkDirections(network, u, s);
        setLinkDirections(network, u, v);
        setLinkDirections(network, t, v);
        break;
    }
    Edge *u_s_edge = getEdgeTo(network, u, s);
    if (u_s_edge->link1->direction == Direction::INCOMING) {
        std::swap(u_s_edge->link1, u_s_edge->link2);
    }
    Edge *u_v_edge = getEdgeTo(network, u, v);
    if (u_v_edge->link1->direction == Direction::INCOMING) {
        std::swap(u_v_edge->link1, u_v_edge->link2);
    }
    Edge *v_t_edge = getEdgeTo(network, v, t);
    if (v_t_edge->link1->direction == Direction::INCOMING) {
        std::swap(v_t_edge->link1, v_t_edge->link2);
    }
}

void checkReticulationProperties(Node *notReticulation, Node *reticulation) {
    if (notReticulation) {
        assert(notReticulation->type == NodeType::BASIC_NODE);
        assert(notReticulation->reticulationData == nullptr);
    }

    if (reticulation) {
        assert(reticulation->type == NodeType::RETICULATION_NODE);
        assert(reticulation->reticulationData->link_to_first_parent);
        assert(reticulation->reticulationData->link_to_second_parent);
        assert(reticulation->reticulationData->link_to_child);
    }
}

void checkLinkDirections(Network &network) {
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        unsigned int targetOutgoing = 2;
        if (network.nodes[i].type == NodeType::RETICULATION_NODE) {
            targetOutgoing = 1;
        } else if (network.root == &network.nodes[i]) {
            targetOutgoing = 2;
        } else if (network.nodes[i].isTip()) {
            targetOutgoing = 0;
        }
        unsigned int n_out = 0;
        for (size_t j = 0; j < network.nodes[i].links.size(); ++j) {
            if (network.nodes[i].links[j].direction == Direction::OUTGOING) {
                n_out++;
            }
        }
        assert(n_out == targetOutgoing);
    }
}

void assertBeforeMove(Network &network, RNNIMove &move) {
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *notReticulation = nullptr;
    Node *reticulation = nullptr;
    if (move.type == RNNIMoveType::ONE_STAR) {
        notReticulation = u;
        reticulation = v;
    } else if (move.type == RNNIMoveType::TWO_STAR) {
        notReticulation = u;
        reticulation = v;
    } else if (move.type == RNNIMoveType::THREE) {
        notReticulation = v;
        reticulation = u;
    } else if (move.type == RNNIMoveType::FOUR) {
        notReticulation = u;
        reticulation = v;
    }
    checkReticulationProperties(notReticulation, reticulation);
    checkLinkDirections(network);
}

void assertAfterMove(Network &network, RNNIMove &move) {
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *notReticulation = nullptr;
    Node *reticulation = nullptr;
    if (move.type == RNNIMoveType::ONE_STAR) {
        notReticulation = v;
        reticulation = u;
    } else if (move.type == RNNIMoveType::TWO_STAR) {
        notReticulation = v;
        reticulation = u;
    } else if (move.type == RNNIMoveType::THREE) {
        notReticulation = u;
        reticulation = v;
    } else if (move.type == RNNIMoveType::FOUR) {
        notReticulation = v;
        reticulation = u;
    }
    checkReticulationProperties(notReticulation, reticulation);
    checkLinkDirections(network);
}

void invalidateLostMegablobRoots(AnnotatedNetwork &ann_network,
        const std::vector<Node*> &previous_megablob_roots) {
    //std::cout << exportDebugInfoBlobs(ann_network.network, ann_network.blobInfo) << "\n";
    std::unordered_set<Node*> act_megablob_roots;
    for (Node *node : ann_network.blobInfo.megablob_roots) {
        act_megablob_roots.emplace(node);
    }
    pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
    //std::vector<bool> visited(ann_network.network.nodes.size(), false);
    for (Node *prevMegablobRoot : previous_megablob_roots) {
        if (act_megablob_roots.find(prevMegablobRoot) == act_megablob_roots.end()) {
            // we need to invalidate that clv
            for (size_t p = 0; p < treeinfo->partition_count; ++p) {
                treeinfo->clv_valid[p][prevMegablobRoot->clv_index] = 0;
            }
            //invalidateHigherCLVs(ann_network, prevMegablobRoot, true, visited);
        }
    }
}

void performMove(AnnotatedNetwork &ann_network, RNNIMove &move) {
    Network &network = ann_network.network;
    std::vector<Node*> previous_megablob_roots = ann_network.blobInfo.megablob_roots;
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    assertBeforeMove(network, move);
    exchangeEdges(network, u, v, s, t);
    updateLinkDirections(network, move);
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE || move.type == RNNIMoveType::FOUR) {
        switchReticulations(network, u, v);
    }
    fixReticulationLinks(u, v, s, t);
    assertAfterMove(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, u, true, visited);
    invalidateHigherCLVs(ann_network, v, true, visited);
    invalidateHigherCLVs(ann_network, s, true, visited);
    invalidateHigherCLVs(ann_network, t, true, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    assertReticulationProbs(ann_network);
    invalidateLostMegablobRoots(ann_network, previous_megablob_roots);
}

void undoMove(AnnotatedNetwork &ann_network, RNNIMove &move) {
    Network &network = ann_network.network;
    std::vector<Node*> previous_megablob_roots = ann_network.blobInfo.megablob_roots;
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    assertAfterMove(network, move);
    exchangeEdges(network, u, v, t, s); // note that s and t are exchanged here
    updateLinkDirectionsReverse(network, move);
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE || move.type == RNNIMoveType::FOUR) {
        switchReticulations(network, u, v);
    }
    fixReticulationLinks(u, v, s, t);
    assertBeforeMove(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, u, true, visited);
    invalidateHigherCLVs(ann_network, v, true, visited);
    invalidateHigherCLVs(ann_network, s, true, visited);
    invalidateHigherCLVs(ann_network, t, true, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    assertReticulationProbs(ann_network);
    invalidateLostMegablobRoots(ann_network, previous_megablob_roots);
}

std::vector<std::pair<Node*, Node*> > getZYChoices(Network &network, Node *x_prime, Node *y_prime,
        Node *x, Node *fixed_y = nullptr, bool returnHead = true, bool returnTail = true) {
    std::vector<std::pair<Node*, Node*> > res;
    auto x_prime_children = getChildren(network, x_prime);
    auto x_children = getChildren(network, x);
    for (Node *z : x_children) {
        if (std::find(x_prime_children.begin(), x_prime_children.end(), z)
                != x_prime_children.end()) {
            continue;
        }
        if (!returnHead && z->type == NodeType::RETICULATION_NODE) { // head-moving rSPR move
            continue;
        }
        if (!returnTail && z->type != NodeType::RETICULATION_NODE) { // tail-moving rSPR move
            continue;
        }
        auto z_children = getChildren(network, z);
        if (std::find(z_children.begin(), z_children.end(), y_prime) != z_children.end()) {
            continue;
        }

        for (Node *y : z_children) {
            if (fixed_y && y != fixed_y) {
                continue;
            }
            if (std::find(x_children.begin(), x_children.end(), y) != x_children.end()) {
                continue;
            }
            assert(hasNeighbor(x, z));
            assert(hasNeighbor(z, y));
            res.emplace_back(std::make_pair(z, y));
        }
    }
    return res;
}

RSPRMove buildRSPRMove(size_t x_prime_clv_index, size_t y_prime_clv_index, size_t x_clv_index,
        size_t y_clv_index, size_t z_clv_index, MoveType moveType) {
    RSPRMove move = RSPRMove();
    move.x_prime_clv_index = x_prime_clv_index;
    move.y_prime_clv_index = y_prime_clv_index;
    move.x_clv_index = x_clv_index;
    move.y_clv_index = y_clv_index;
    move.z_clv_index = z_clv_index;
    move.moveType = moveType;
    return move;
}

void possibleRSPRMovesInternal(std::vector<RSPRMove> &res, AnnotatedNetwork &ann_network, Node *x_prime,
        Node *y_prime, Node *x, Node *fixed_y, bool returnHead, bool returnTail,
        MoveType moveType) {
    Network &network = ann_network.network;
    auto zy = getZYChoices(network, x_prime, y_prime, x, fixed_y, returnHead, returnTail);
    for (const auto &entry : zy) {
        Node *z = entry.first;
        Node *y = entry.second;

        Node *w = nullptr;
        auto zNeighbors = getNeighbors(network, z);
        assert(zNeighbors.size() == 3);
        for (size_t j = 0; j < zNeighbors.size(); ++j) {
            if (zNeighbors[j] != x && zNeighbors[j] != y) {
                w = zNeighbors[j];
                break;
            }
        }
        assert(w);

        if (z->type == NodeType::RETICULATION_NODE) { // head-moving rSPR move
            if (!hasPath(network, y_prime, w)) {
                RSPRMove move = buildRSPRMove(x_prime->clv_index, y_prime->clv_index, x->clv_index,
                        y->clv_index, z->clv_index, moveType);
                move.x_z_len = get_edge_lengths(ann_network, getEdgeTo(network, x, z)->pmatrix_index);
                move.z_y_len = get_edge_lengths(ann_network, getEdgeTo(network, z, y)->pmatrix_index);
                move.x_prime_y_prime_len = get_edge_lengths(ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
                res.emplace_back(move);
            }
        } else { // tail-moving rSPR move
            if (!hasPath(network, w, x_prime)) {
                RSPRMove move = buildRSPRMove(x_prime->clv_index, y_prime->clv_index, x->clv_index,
                        y->clv_index, z->clv_index, moveType);
                move.x_z_len = get_edge_lengths(ann_network, getEdgeTo(network, x, z)->pmatrix_index);
                move.z_y_len = get_edge_lengths(ann_network, getEdgeTo(network, z, y)->pmatrix_index);
                move.x_prime_y_prime_len = get_edge_lengths(ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
                res.emplace_back(move);
            }
        }
    }
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge,
        Node *fixed_x, Node *fixed_y, MoveType moveType, bool returnHead = true, bool returnTail =
                true) {
    Network &network = ann_network.network;
    std::vector<RSPRMove> res;
    Node *x_prime = getSource(network, edge);
    Node *y_prime = getTarget(network, edge);

    if (fixed_x) {
        possibleRSPRMovesInternal(res, ann_network, x_prime, y_prime, fixed_x, fixed_y, returnHead,
                returnTail, moveType);
    } else {
        for (size_t i = 0; i < network.num_nodes(); ++i) {
            Node *x = &network.nodes[i];
            possibleRSPRMovesInternal(res, ann_network, x_prime, y_prime, x, fixed_y, returnHead,
                    returnTail, moveType);
        }
    }
    return res;
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge) {
    return possibleRSPRMoves(ann_network, edge, nullptr, nullptr, MoveType::RSPRMove, true, true);
}

std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, const Edge *edge) {
    return possibleRSPRMoves(ann_network, edge, nullptr, nullptr, MoveType::TailMove, false, true);
}

std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, const Edge *edge) {
    return possibleRSPRMoves(ann_network, edge, nullptr, nullptr, MoveType::HeadMove, true, false);
}

std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleTailMoves(ann_network, &network.edges[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleHeadMoves(ann_network, &network.edges[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, const Edge *edge) {
    Network &network = ann_network.network;
// in an rSPR1 move, either y_prime == x, x_prime == y, x_prime == x, or y_prime == y
    std::vector<RSPRMove> res;
    Node *x_prime = getSource(network, edge);
    Node *y_prime = getTarget(network, edge);

// Case 1: y_prime == x
    std::vector<RSPRMove> case1 = possibleRSPRMoves(ann_network, edge, y_prime, nullptr,
            MoveType::RSPR1Move);
    res.insert(std::end(res), std::begin(case1), std::end(case1));

// Case 2: x_prime == x
    std::vector<RSPRMove> case2 = possibleRSPRMoves(ann_network, edge, x_prime, nullptr,
            MoveType::RSPR1Move);
    res.insert(std::end(res), std::begin(case2), std::end(case2));

// Case 3: x_prime == y
    std::vector<RSPRMove> case3 = possibleRSPRMoves(ann_network, edge, nullptr, x_prime,
            MoveType::RSPR1Move);
    res.insert(std::end(res), std::begin(case3), std::end(case3));

// Case 4: y_prime == y
    std::vector<RSPRMove> case4 = possibleRSPRMoves(ann_network, edge, nullptr, y_prime,
            MoveType::RSPR1Move);
    res.insert(std::end(res), std::begin(case4), std::end(case4));

    return res;
}

std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network) {
    std::vector<RNNIMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RNNIMove> branch_moves = possibleRNNIMoves(ann_network, &network.edges[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleRSPRMoves(ann_network, &network.edges[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleRSPR1Moves(ann_network, &network.edges[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

ArcInsertionMove buildArcInsertionMove(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
        std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType) {
    ArcInsertionMove move = ArcInsertionMove();
    move.a_clv_index = a_clv_index;
    move.b_clv_index = b_clv_index;
    move.c_clv_index = c_clv_index;
    move.d_clv_index = d_clv_index;

    move.u_v_len = u_v_len;
    move.c_v_len = c_v_len;
    move.a_u_len = a_u_len;

    move.a_b_len = a_b_len;
    move.c_d_len = c_d_len;
    move.v_d_len = v_d_len;
    move.u_b_len = u_b_len;

    move.moveType = moveType;
    return move;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const Edge *edge, Node *c, Node *d, MoveType moveType) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    size_t n_p = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1);
// choose two distinct arcs ab, cd (with cd not ancestral to ab -> no d-a-path allowed)
    Node *a = getSource(network, edge);
    Node *b = getTarget(network, edge);
    std::vector<double> a_b_len(n_p);
    for (size_t p = 0; p < n_p; ++p) {
        a_b_len[p] = ann_network.fake_treeinfo->branch_lengths[p][edge->pmatrix_index];
    }

    double min_br = ann_network.options.brlen_min;

    Node *c_cand = nullptr;
    Node *d_cand = nullptr;
    if (c) {
        c_cand = c;
        for (size_t i = 0; i < c->links.size(); ++i) {
            if (c->links[i].direction == Direction::INCOMING) {
                continue;
            }
            Node *d_cand = network.nodes_by_index[c->links[i].outer->node_clv_index];
            if (a->clv_index == c_cand->clv_index && b->clv_index == d_cand->clv_index) {
                continue;
            }
            if (!hasPath(network, d_cand, a)) {
                std::vector<double> c_d_len(n_p), c_v_len(n_p), a_u_len(n_p), v_d_len(n_p), u_b_len(n_p), u_v_len(n_p);
                for (size_t p = 0; p < n_p; ++p) {
                    c_d_len[p] = ann_network.fake_treeinfo->branch_lengths[p][c->links[i].edge_pmatrix_index];

                    c_v_len[p] = std::max(c_d_len[p] / 2, min_br);
                    a_u_len[p] = std::max(a_b_len[p] / 2, min_br);
                    v_d_len[p] = std::max(c_d_len[p] - c_v_len[p], min_br);
                    u_b_len[p] = std::max(a_b_len[p] - a_u_len[p], min_br);
                    u_v_len[p] = 1.0;
                }

                ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                        c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                        a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType);
                move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                res.emplace_back(move);
            }
        }
    } else if (d) {
        d_cand = d;
        if (!hasPath(network, d_cand, a)) {
            for (size_t i = 0; i < d->links.size(); ++i) {
                if (d->links[i].direction == Direction::OUTGOING) {
                    continue;
                }
                Node *c_cand = network.nodes_by_index[d->links[i].outer->node_clv_index];
                if (a->clv_index == c_cand->clv_index && b->clv_index == d_cand->clv_index) {
                    continue;
                }
                std::vector<double> c_d_len(n_p), c_v_len(n_p), a_u_len(n_p), v_d_len(n_p), u_b_len(n_p), u_v_len(n_p);
                for (size_t p = 0; p < n_p; ++p) {
                    c_d_len[p] = ann_network.fake_treeinfo->branch_lengths[p][d->links[i].edge_pmatrix_index];

                    c_v_len[p] = std::max(c_d_len[p] / 2, min_br);
                    a_u_len[p] = std::max(a_b_len[p] / 2, min_br);
                    v_d_len[p] = std::max(c_d_len[p] - c_v_len[p], min_br);
                    u_b_len[p] = std::max(a_b_len[p] - a_u_len[p], min_br);
                    u_v_len[p] = 1.0;
                }

                ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                        c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                        a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType);

                move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                res.emplace_back(move);
            }
        }
    } else {
        for (size_t i = 0; i < network.num_branches(); ++i) {
            if (network.edges[i].pmatrix_index == edge->pmatrix_index) {
                continue;
            }
            Node *c_cand = getSource(network, &network.edges[i]);
            Node *d_cand = getTarget(network, &network.edges[i]);
            if (!hasPath(network, d_cand, a)) {
                std::vector<double> c_d_len(n_p), c_v_len(n_p), a_u_len(n_p), v_d_len(n_p), u_b_len(n_p), u_v_len(n_p);
                for (size_t p = 0; p < n_p; ++p) {
                    c_d_len[p] = ann_network.fake_treeinfo->branch_lengths[p][network.edges[i].pmatrix_index];

                    c_v_len[p] = std::max(c_d_len[p] / 2, min_br);
                    a_u_len[p] = std::max(a_b_len[p] / 2, min_br);
                    v_d_len[p] = std::max(c_d_len[p] - c_v_len[p], min_br);
                    u_b_len[p] = std::max(a_b_len[p] - a_u_len[p], min_br);
                    u_v_len[p] = 1.0;
                }

                ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                        c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                        a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType);

                move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                res.emplace_back(move);
            }
        }
    }
    return res;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const Edge *edge) {
    return possibleArcInsertionMoves(ann_network, edge, nullptr, nullptr,
            MoveType::ArcInsertionMove);
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network,
        const Edge *edge) {
    Network &network = ann_network.network;
    std::vector<ArcInsertionMove> res;
    Node *a = getSource(network, edge);
    Node *b = getTarget(network, edge);

// Case 1: a == c
    std::vector<ArcInsertionMove> case1 = possibleArcInsertionMoves(ann_network, edge, a, nullptr,
            MoveType::DeltaPlusMove);
    res.insert(std::end(res), std::begin(case1), std::end(case1));

// Case 2: b == d
    std::vector<ArcInsertionMove> case2 = possibleArcInsertionMoves(ann_network, edge, nullptr, b,
            MoveType::DeltaPlusMove);
    res.insert(std::end(res), std::begin(case2), std::end(case2));

// Case 3: b == c
    std::vector<ArcInsertionMove> case3 = possibleArcInsertionMoves(ann_network, edge, b, nullptr,
            MoveType::DeltaPlusMove);
    res.insert(std::end(res), std::begin(case3), std::end(case3));

    return res;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<ArcInsertionMove> moves = possibleArcInsertionMoves(ann_network,
                &network.edges[i], nullptr, nullptr, MoveType::ArcInsertionMove);
        res.insert(std::end(res), std::begin(moves), std::end(moves));
    }
    return res;
}

ArcRemovalMove buildArcRemovalMove(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, size_t u_clv_index, size_t v_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
         std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType) {
    ArcRemovalMove move = ArcRemovalMove();
    move.a_clv_index = a_clv_index;
    move.b_clv_index = b_clv_index;
    move.c_clv_index = c_clv_index;
    move.d_clv_index = d_clv_index;
    move.u_clv_index = u_clv_index;
    move.v_clv_index = v_clv_index;

    move.u_v_len = u_v_len;
    move.c_v_len = c_v_len;
    move.a_u_len = a_u_len;
    
    move.a_b_len = a_b_len;
    move.c_d_len = c_d_len;
    move.v_d_len = v_d_len;
    move.u_b_len = u_b_len;
    move.moveType = moveType;
    return move;
}

std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network, Node *v,
        MoveType moveType) {
// v is a reticulation node, u is one parent of v, c is the other parent of v, a is parent of u, d is child of v, b is other child of u
    std::vector<ArcRemovalMove> res;
    Network &network = ann_network.network;
    size_t n_p = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1);
    assert(v->type == NodeType::RETICULATION_NODE);
    Node *d = getReticulationChild(network, v);
    Node *first_parent = getReticulationFirstParent(network, v);
    Node *second_parent = getReticulationSecondParent(network, v);
    std::vector<std::pair<Node*, Node*> > ucChoices;
    double max_br = ann_network.options.brlen_max;
    if (first_parent != network.root && first_parent->type != NodeType::RETICULATION_NODE) {
        ucChoices.emplace_back(std::make_pair(first_parent, second_parent));
    }
    if (second_parent != network.root && second_parent->type != NodeType::RETICULATION_NODE) {
        ucChoices.emplace_back(std::make_pair(second_parent, first_parent));
    }
    for (size_t i = 0; i < ucChoices.size(); ++i) {
        Node *u = ucChoices[i].first;
        Node *c = ucChoices[i].second;
        Node *b = getOtherChild(network, u, v);
        if (hasChild(network, c, d)) { // avoid creating parallel arcs
            continue;
        }
        assert(u);
        assert(c);
        assert(b);
        Node *a = getActiveParent(network, u);
        assert(a);
        if (hasChild(network, a, b)) { // avoid creating parallel arcs
            continue;
        }

        if (a->clv_index == c->clv_index && b->clv_index == d->clv_index) {
            continue;
        }

        std::vector<double> a_u_len(n_p), u_b_len(n_p), a_b_len(n_p), c_v_len(n_p), v_d_len(n_p), c_d_len(n_p), u_v_len(n_p);

        for (size_t p = 0; p < n_p; ++p) {
            a_u_len[p] = ann_network.fake_treeinfo->branch_lengths[p][getEdgeTo(network, a, u)->pmatrix_index];
            u_b_len[p] = ann_network.fake_treeinfo->branch_lengths[p][getEdgeTo(network, u, b)->pmatrix_index];
            a_b_len[p] = std::min(a_u_len[p] + u_b_len[p], max_br);

            c_v_len[p] = ann_network.fake_treeinfo->branch_lengths[p][getEdgeTo(network, c, v)->pmatrix_index];
            v_d_len[p] = ann_network.fake_treeinfo->branch_lengths[p][getEdgeTo(network, v, d)->pmatrix_index];
            c_d_len[p] = std::min(c_v_len[p] + v_d_len[p], max_br);

            u_v_len[p] = ann_network.fake_treeinfo->branch_lengths[p][getEdgeTo(network, u, v)->pmatrix_index];
        }

        ArcRemovalMove move = buildArcRemovalMove(a->clv_index, b->clv_index, c->clv_index,
                d->clv_index, u->clv_index, v->clv_index, u_v_len,
                c_v_len, a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType);

        move.au_pmatrix_index = getEdgeTo(network, a, u)->pmatrix_index;
        move.ub_pmatrix_index = getEdgeTo(network, u, b)->pmatrix_index;
        move.cv_pmatric_index = getEdgeTo(network, c, v)->pmatrix_index;
        move.vd_pmatrix_index = getEdgeTo(network, v, d)->pmatrix_index;
        move.uv_pmatrix_index = getEdgeTo(network, u, v)->pmatrix_index;

        res.emplace_back(move);
    }
    return res;
}

std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network, Node *v) {
    std::vector<ArcRemovalMove> res;
    std::vector<ArcRemovalMove> allRemovals = possibleArcRemovalMoves(ann_network, v,
            MoveType::DeltaMinusMove);
// 3 cases: a == c, b == d, or b == c
    for (size_t i = 0; i < allRemovals.size(); ++i) {
        if ((allRemovals[i].a_clv_index == allRemovals[i].c_clv_index)
                || (allRemovals[i].b_clv_index == allRemovals[i].d_clv_index)
                || (allRemovals[i].b_clv_index == allRemovals[i].c_clv_index)) {
            res.emplace_back(allRemovals[i]);
        }
    }
    return res;
}

std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network) {
    std::vector<ArcRemovalMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        auto moves = possibleArcRemovalMoves(ann_network, network.reticulation_nodes[i],
                MoveType::ArcRemovalMove);
        res.insert(std::end(res), std::begin(moves), std::end(moves));
    }
    return res;
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<ArcInsertionMove> branch_moves = possibleDeltaPlusMoves(ann_network,
                &network.edges[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network) {
    std::vector<ArcRemovalMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        std::vector<ArcRemovalMove> branch_moves = possibleDeltaMinusMoves(ann_network,
                network.reticulation_nodes[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

void performMove(AnnotatedNetwork &ann_network, RSPRMove &move) {
    Network &network = ann_network.network;
    std::vector<Node*> previous_megablob_roots = ann_network.blobInfo.megablob_roots;
    Node *x_prime = network.nodes_by_index[move.x_prime_clv_index];
    Node *y_prime = network.nodes_by_index[move.y_prime_clv_index];
    Node *x = network.nodes_by_index[move.x_clv_index];
    Node *y = network.nodes_by_index[move.y_clv_index];
    Node *z = network.nodes_by_index[move.z_clv_index];

    Link *x_out_link = getLinkToNode(network, x, z);
    Link *z_in_link = getLinkToNode(network, z, x);
    Link *z_out_link = getLinkToNode(network, z, y);
    Link *x_prime_out_link = getLinkToNode(network, x_prime, y_prime);
    Link *y_prime_in_link = getLinkToNode(network, y_prime, x_prime);
    Link *y_in_link = getLinkToNode(network, y, z);

    Edge *x_z_edge = getEdgeTo(network, x, z);
    Edge *z_y_edge = getEdgeTo(network, z, y);
    Edge *x_prime_y_prime_edge = getEdgeTo(network, x_prime, y_prime);

    std::vector<double> x_z_len = get_edge_lengths(ann_network, x_z_edge->pmatrix_index);
    std::vector<double> z_y_len = get_edge_lengths(ann_network, z_y_edge->pmatrix_index);
    std::vector<double> x_prime_y_prime_len = get_edge_lengths(ann_network, x_prime_y_prime_edge->pmatrix_index);

    double min_br = ann_network.options.brlen_min;
    double max_br = ann_network.options.brlen_max;

    size_t n_p = (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) ? ann_network.fake_treeinfo->partition_count : 1;
    std::vector<double> x_y_len(n_p), x_prime_z_len(n_p), z_y_prime_len(n_p);
    for (size_t p = 0; p < n_p; ++p) {
        x_y_len[p] = std::min(x_z_len[p] + z_y_len[p], max_br);
        assert(x_y_len[p] >= min_br);
        x_prime_z_len[p] = std::max(x_prime_y_prime_len[p] / 2, min_br);
        assert(x_prime_z_len[p] <= max_br);
        z_y_prime_len[p] = std::max(x_prime_y_prime_len[p] / 2, min_br);
        assert(x_prime_y_prime_len[p] <= max_br);
    }

    assert(x_prime_out_link->edge_pmatrix_index == x_prime_y_prime_edge->pmatrix_index);
    assert(y_prime_in_link->edge_pmatrix_index == x_prime_y_prime_edge->pmatrix_index);
    assert(x_out_link->edge_pmatrix_index == x_z_edge->pmatrix_index);
    assert(z_in_link->edge_pmatrix_index == x_z_edge->pmatrix_index);
    assert(z_out_link->edge_pmatrix_index == z_y_edge->pmatrix_index);
    assert(y_in_link->edge_pmatrix_index == z_y_edge->pmatrix_index);

    x_out_link->outer = y_in_link;
    y_in_link->outer = x_out_link;
    x_prime_out_link->outer = z_in_link;
    z_in_link->outer = x_prime_out_link;
    z_out_link->outer = y_prime_in_link;
    y_prime_in_link->outer = z_out_link;

    Edge *x_y_edge = x_prime_y_prime_edge;
    Edge *x_prime_z_edge = x_z_edge;
    Edge *z_y_prime_edge = z_y_edge;
    x_y_edge->link1 = x_out_link;
    x_y_edge->link2 = y_in_link;
    x_prime_z_edge->link1 = x_prime_out_link;
    x_prime_z_edge->link2 = z_in_link;
    z_y_prime_edge->link1 = z_out_link;
    z_y_prime_edge->link2 = y_prime_in_link;

    set_edge_lengths(ann_network, x_y_edge->pmatrix_index, x_y_len);
    set_edge_lengths(ann_network, x_prime_z_edge->pmatrix_index, x_prime_z_len);
    set_edge_lengths(ann_network, z_y_prime_edge->pmatrix_index, z_y_prime_len);

    x_out_link->edge_pmatrix_index = x_y_edge->pmatrix_index;
    y_in_link->edge_pmatrix_index = x_y_edge->pmatrix_index;
    x_prime_out_link->edge_pmatrix_index = x_prime_z_edge->pmatrix_index;
    z_in_link->edge_pmatrix_index = x_prime_z_edge->pmatrix_index;
    z_out_link->edge_pmatrix_index = z_y_prime_edge->pmatrix_index;
    y_prime_in_link->edge_pmatrix_index = z_y_prime_edge->pmatrix_index;

    fixReticulations(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, z, false, visited);
    invalidateHigherCLVs(ann_network, x, false, visited);
    invalidateHigherCLVs(ann_network, x_prime, false, visited);
    invalidateHigherCLVs(ann_network, y, false, visited);
    invalidateHigherCLVs(ann_network, y_prime, false, visited);
    invalidatePmatrixIndex(ann_network, x_y_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, x_prime_z_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, z_y_prime_edge->pmatrix_index, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);

    //std::cout << exportDebugInfo(ann_network.network) << "\n";

    assertReticulationProbs(ann_network);
    invalidateLostMegablobRoots(ann_network, previous_megablob_roots);
}

void undoMove(AnnotatedNetwork &ann_network, RSPRMove &move) {
    Network &network = ann_network.network;
    std::vector<Node*> previous_megablob_roots = ann_network.blobInfo.megablob_roots;
    Node *x_prime = network.nodes_by_index[move.x_prime_clv_index];
    Node *y_prime = network.nodes_by_index[move.y_prime_clv_index];
    Node *x = network.nodes_by_index[move.x_clv_index];
    Node *y = network.nodes_by_index[move.y_clv_index];
    Node *z = network.nodes_by_index[move.z_clv_index];

    Link *x_out_link = getLinkToNode(network, x, y);
    Link *z_in_link = getLinkToNode(network, z, x_prime);
    Link *z_out_link = getLinkToNode(network, z, y_prime);
    Link *x_prime_out_link = getLinkToNode(network, x_prime, z);
    Link *y_prime_in_link = getLinkToNode(network, y_prime, z);
    Link *y_in_link = getLinkToNode(network, y, x);

    Edge *x_y_edge = getEdgeTo(network, x, y);
    Edge *x_prime_z_edge = getEdgeTo(network, x_prime, z);
    Edge *z_y_prime_edge = getEdgeTo(network, z, y_prime);

    std::vector<double> x_z_len = move.x_z_len;
    std::vector<double> z_y_len = move.z_y_len;
    std::vector<double> x_prime_y_prime_len = move.x_prime_y_prime_len;

    assert(x_out_link->edge_pmatrix_index == x_y_edge->pmatrix_index);
    assert(y_in_link->edge_pmatrix_index == x_y_edge->pmatrix_index);
    assert(x_prime_out_link->edge_pmatrix_index == x_prime_z_edge->pmatrix_index);
    assert(z_in_link->edge_pmatrix_index == x_prime_z_edge->pmatrix_index);
    assert(z_out_link->edge_pmatrix_index == z_y_prime_edge->pmatrix_index);
    assert(y_prime_in_link->edge_pmatrix_index == z_y_prime_edge->pmatrix_index);

    x_out_link->outer = z_in_link;
    z_in_link->outer = x_out_link;
    x_prime_out_link->outer = y_prime_in_link;
    y_prime_in_link->outer = x_prime_out_link;
    z_out_link->outer = y_in_link;
    y_in_link->outer = z_out_link;

    Edge *x_prime_y_prime_edge = x_y_edge;
    Edge *x_z_edge = x_prime_z_edge;
    Edge *z_y_edge = z_y_prime_edge;
    x_prime_y_prime_edge->link1 = x_prime_out_link;
    x_prime_y_prime_edge->link2 = y_prime_in_link;
    x_z_edge->link1 = x_out_link;
    x_z_edge->link2 = z_in_link;
    z_y_edge->link1 = z_out_link;
    z_y_edge->link2 = y_in_link;

    set_edge_lengths(ann_network, x_prime_y_prime_edge->pmatrix_index, x_prime_y_prime_len);
    set_edge_lengths(ann_network, x_z_edge->pmatrix_index, x_z_len);
    set_edge_lengths(ann_network, z_y_edge->pmatrix_index, z_y_len);

    x_prime_out_link->edge_pmatrix_index = x_prime_y_prime_edge->pmatrix_index;
    y_prime_in_link->edge_pmatrix_index = x_prime_y_prime_edge->pmatrix_index;
    x_out_link->edge_pmatrix_index = x_z_edge->pmatrix_index;
    z_in_link->edge_pmatrix_index = x_z_edge->pmatrix_index;
    z_out_link->edge_pmatrix_index = z_y_edge->pmatrix_index;
    y_in_link->edge_pmatrix_index = z_y_edge->pmatrix_index;

    fixReticulations(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, z, false, visited);
    invalidateHigherCLVs(ann_network, x, false, visited);
    invalidateHigherCLVs(ann_network, x_prime, false, visited);
    invalidateHigherCLVs(ann_network, y, false, visited);
    invalidateHigherCLVs(ann_network, y_prime, false, visited);
    invalidatePmatrixIndex(ann_network, x_prime_y_prime_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, x_z_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, z_y_edge->pmatrix_index, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    assertReticulationProbs(ann_network);
    invalidateLostMegablobRoots(ann_network, previous_megablob_roots);
}

void removeEdge(Network &network, Edge *edge) {
    assert(edge);
    size_t index = edge->pmatrix_index;
    size_t other_index = network.edges[network.branchCount - 1].pmatrix_index;
    size_t index_in_edges_array = network.edges_by_index[index] - &network.edges[0];
    assert(network.edges[index_in_edges_array].pmatrix_index == index);
    std::swap(network.edges[index_in_edges_array], network.edges[network.branchCount - 1]);
    network.edges_by_index[other_index] = &network.edges[index_in_edges_array];
    network.edges_by_index[index] = nullptr;
    network.edges[network.branchCount - 1].clear();
    network.branchCount--;
}

Edge* addEdgeInternal(AnnotatedNetwork &ann_network, Link *link1, Link *link2, std::vector<double> length,
        size_t pmatrix_index) {
    assert(ann_network.network.num_branches() < ann_network.network.edges.size());
    if (link1->direction == Direction::INCOMING) {
        std::swap(link1, link2);
    }

    assert(ann_network.network.edges_by_index[pmatrix_index] == nullptr);
    ann_network.network.edges[ann_network.network.branchCount].init(pmatrix_index, link1, link2, length[0], 1.0);
    set_edge_lengths(ann_network, pmatrix_index, length);
    ann_network.network.edges_by_index[pmatrix_index] = &ann_network.network.edges[ann_network.network.branchCount];
    ann_network.network.branchCount++;

    return ann_network.network.edges_by_index[pmatrix_index];
}

Edge* addEdge(AnnotatedNetwork &ann_network, Link *link1, Link *link2, std::vector<double> length,
        size_t wanted_pmatrix_index) {
    if (link1->direction == Direction::INCOMING) {
        std::swap(link1, link2);
    }
    size_t pmatrix_index = 0;
    if (wanted_pmatrix_index < ann_network.network.edges.size() && ann_network.network.edges_by_index[wanted_pmatrix_index] == nullptr) {
        pmatrix_index = wanted_pmatrix_index;
    } else {
        // find smallest free non-tip pmatrix index
        for (size_t i = 0; i < ann_network.network.edges.size(); ++i) {
            if (ann_network.network.edges_by_index[i] == nullptr) {
                pmatrix_index = i;
                break;
            }
        }
    }
    assert(ann_network.network.edges_by_index[pmatrix_index] == nullptr);

    return addEdgeInternal(ann_network, link1, link2, length, pmatrix_index);
}

void checkSanity(Network &network) {
// check edge<->links sanity
    for (size_t i = 0; i < network.edges.size(); ++i) {
        if (network.edges_by_index[i]) {
            assert(network.edges_by_index[i]->link1->edge_pmatrix_index == i);
            assert(network.edges_by_index[i]->link2->edge_pmatrix_index == i);
        }
    }
// check node<->links sanity
    for (size_t i = 0; i < network.nodes.size(); ++i) {
        if (network.nodes_by_index[i]) {
            assert(network.nodes_by_index[i]->links.size() <= 3);
        }
    }
}

void removeNode(Network &network, Node *node) {
    assert(node);
    assert(!node->isTip());
    size_t root_idx = network.root->clv_index;
    size_t index = node->clv_index;
    size_t other_index = network.nodes[network.nodeCount - 1].clv_index;
    size_t index_in_nodes_array = network.nodes_by_index[index] - &network.nodes[0];
    assert(network.nodes[index_in_nodes_array].clv_index == index);
    std::swap(network.nodes[index_in_nodes_array], network.nodes[network.nodeCount - 1]);
    network.nodes_by_index[other_index] = &network.nodes[index_in_nodes_array];
    network.nodes_by_index[index] = &network.nodes[network.nodeCount - 1];
    node = network.nodes_by_index[index];

    if (node->type == NodeType::RETICULATION_NODE) {
        network.reticulation_nodes[node->getReticulationData()->reticulation_index] =
                &network.nodes[network.nodeCount - 1];
    }

    if (network.nodes_by_index[other_index]->type == NodeType::RETICULATION_NODE) {
        network.reticulation_nodes[network.nodes_by_index[other_index]->getReticulationData()->getReticulationIndex()] =
                &network.nodes[index_in_nodes_array];
    }

    if (network.nodes_by_index[other_index]->type == NodeType::RETICULATION_NODE
            && node->type == NodeType::RETICULATION_NODE) {
        unsigned int other_ret_index =
                network.nodes_by_index[other_index]->getReticulationData()->reticulation_index;
        unsigned int node_ret_index = node->getReticulationData()->reticulation_index;
        if (node_ret_index < other_ret_index) {
// swap the reticulation indices
            network.reticulation_nodes[other_ret_index]->getReticulationData()->reticulation_index =
                    node_ret_index;
            network.reticulation_nodes[node_ret_index]->getReticulationData()->reticulation_index =
                    other_ret_index;
            network.reticulation_nodes[other_ret_index] = network.nodes_by_index[index];
            network.reticulation_nodes[node_ret_index] = network.nodes_by_index[other_index];
            std::swap(node_ret_index, other_ret_index);
        }
        network.reticulation_nodes[other_ret_index] = network.nodes_by_index[other_index];
    }

    if (node->type == NodeType::RETICULATION_NODE) {
        if (network.num_reticulations() > 1) {
// update reticulation indices
            unsigned int bad_reticulation_index = node->getReticulationData()->reticulation_index;
            network.reticulation_nodes[network.reticulation_nodes.size() - 1]->getReticulationData()->reticulation_index =
                    bad_reticulation_index;
            std::swap(network.reticulation_nodes[bad_reticulation_index],
                    network.reticulation_nodes[network.reticulation_nodes.size() - 1]);
        }
        network.reticulation_nodes.resize(network.reticulation_nodes.size() - 1);

        for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
            assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
        }
    }
    network.nodes_by_index[index] = nullptr;
    network.nodes[network.nodeCount - 1].clear();
    network.root = network.nodes_by_index[root_idx];
    network.nodeCount--;
}

Node* addInnerNode(Network &network, ReticulationData *retData, size_t wanted_clv_index) {
    assert(network.num_nodes() < network.nodes.size());
    unsigned int clv_index;

    if (wanted_clv_index < network.nodes.size() && network.nodes_by_index[wanted_clv_index] == nullptr) {
        clv_index = wanted_clv_index;
    } else {
        clv_index = network.nodes.size() - 1;
        // try to find a smaller unused clv index
        for (size_t i = 0; i < clv_index; ++i) {
            if (network.nodes_by_index[i] == nullptr) {
                clv_index = i;
                break;
            }
        }
    }
    assert(network.nodes_by_index[clv_index] == nullptr);
    unsigned int scaler_index = clv_index - network.num_tips();
    network.nodes_by_index[clv_index] = &network.nodes[network.nodeCount];

    if (retData) {
        network.nodes[network.nodeCount].initReticulation(clv_index, scaler_index, "", *retData);
        network.reticulation_nodes.emplace_back(network.nodes_by_index[clv_index]);
        network.nodes[network.nodeCount].getReticulationData()->reticulation_index =
                network.reticulation_nodes.size() - 1;
        for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
            assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
        }
    } else {
        network.nodes[network.nodeCount].initBasic(clv_index, scaler_index, "");
    }

    network.nodeCount++;
    return network.nodes_by_index[clv_index];
}

void invalidate_pmatrices(AnnotatedNetwork &ann_network,
        std::vector<size_t> &affectedPmatrixIndices) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t *fake_treeinfo = ann_network.fake_treeinfo;
    unsigned int partitions = 1;
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) { // each partition has its branch length
        partitions = fake_treeinfo->partition_count;
    }
    for (size_t pmatrix_index : affectedPmatrixIndices) {
        assert(network.edges_by_index[pmatrix_index]);
        for (size_t p = 0; p < partitions; ++p) {
            fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
    }
    pllmod_treeinfo_update_prob_matrices(fake_treeinfo, 0);
}

void performMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move) {
    Network &network = ann_network.network;
    std::vector<Node*> previous_megablob_roots = ann_network.blobInfo.megablob_roots;

    Link *from_a_link = getLinkToNode(network, move.a_clv_index, move.b_clv_index);
    Link *to_b_link = getLinkToNode(network, move.b_clv_index, move.a_clv_index);
    Link *from_c_link = getLinkToNode(network, move.c_clv_index, move.d_clv_index);
    Link *to_d_link = getLinkToNode(network, move.d_clv_index, move.c_clv_index);
    Edge *a_b_edge = getEdgeTo(network, move.a_clv_index, move.b_clv_index);
    assert(a_b_edge->link1);
    assert(a_b_edge->link2);
    Edge *c_d_edge = getEdgeTo(network, move.c_clv_index, move.d_clv_index);
    assert(c_d_edge->link1);
    assert(c_d_edge->link2);
    size_t a_b_edge_index = a_b_edge->pmatrix_index;
    size_t c_d_edge_index = c_d_edge->pmatrix_index;

    ReticulationData retData;
    retData.init(network.num_reticulations(), "", 0, nullptr, nullptr, nullptr);
    Node *u = addInnerNode(network, nullptr, move.wanted_u_clv_index);
    Node *v = addInnerNode(network, &retData, move.wanted_v_clv_index);

    move.wanted_u_clv_index = u->clv_index;
    move.wanted_v_clv_index = v->clv_index;

    Link *to_u_link = make_link(u, nullptr, Direction::INCOMING);
    Link *u_b_link = make_link(u, nullptr, Direction::OUTGOING);
    Link *u_v_link = make_link(u, nullptr, Direction::OUTGOING);

    Link *v_u_link = make_link(v, nullptr, Direction::INCOMING);
    Link *v_c_link = make_link(v, nullptr, Direction::INCOMING);
    Link *v_d_link = make_link(v, nullptr, Direction::OUTGOING);

    std::vector<double> u_v_edge_length = move.u_v_len;
    std::vector<double> c_v_edge_length = move.c_v_len;
    std::vector<double> v_d_edge_length = move.v_d_len;
    std::vector<double> a_u_edge_length = move.a_u_len;
    std::vector<double> u_b_edge_length = move.u_b_len;

    removeEdge(network, network.edges_by_index[a_b_edge_index]);
    if (c_d_edge_index != a_b_edge_index) {
        removeEdge(network, network.edges_by_index[c_d_edge_index]);
    }

    Edge *u_b_edge = addEdge(ann_network, u_b_link, to_b_link, u_b_edge_length,
            move.wanted_ub_pmatrix_index);
    Edge *v_d_edge = addEdge(ann_network, v_d_link, to_d_link, v_d_edge_length,
            move.wanted_vd_pmatrix_index);
    Edge *a_u_edge = addEdge(ann_network, from_a_link, to_u_link, a_u_edge_length,
            move.wanted_au_pmatrix_index);
    Edge *c_v_edge = addEdge(ann_network, from_c_link, v_c_link, c_v_edge_length,
            move.wanted_cv_pmatrix_index);
    Edge *u_v_edge = addEdge(ann_network, u_v_link, v_u_link, u_v_edge_length,
            move.wanted_uv_pmatrix_index);

    v->getReticulationData()->link_to_first_parent = v_u_link;
    v->getReticulationData()->link_to_second_parent = v_c_link;
    v->getReticulationData()->link_to_child = v_d_link;
    if (v->getReticulationData()->link_to_first_parent->edge_pmatrix_index
            > v->getReticulationData()->link_to_second_parent->edge_pmatrix_index) {
        std::swap(v->getReticulationData()->link_to_first_parent,
                v->getReticulationData()->link_to_second_parent);
    }

    from_a_link->edge_pmatrix_index = a_u_edge->pmatrix_index;
    to_u_link->edge_pmatrix_index = a_u_edge->pmatrix_index;
    u_b_link->edge_pmatrix_index = u_b_edge->pmatrix_index;
    to_b_link->edge_pmatrix_index = u_b_edge->pmatrix_index;
    u_v_link->edge_pmatrix_index = u_v_edge->pmatrix_index;
    v_u_link->edge_pmatrix_index = u_v_edge->pmatrix_index;
    v_c_link->edge_pmatrix_index = c_v_edge->pmatrix_index;
    from_c_link->edge_pmatrix_index = c_v_edge->pmatrix_index;
    v_d_link->edge_pmatrix_index = v_d_edge->pmatrix_index;
    to_d_link->edge_pmatrix_index = v_d_edge->pmatrix_index;

    from_a_link->outer = to_u_link;
    to_u_link->outer = from_a_link;
    u_b_link->outer = to_b_link;
    to_b_link->outer = u_b_link;
    u_v_link->outer = v_u_link;
    v_u_link->outer = u_v_link;
    v_c_link->outer = from_c_link;
    from_c_link->outer = v_c_link;
    v_d_link->outer = to_d_link;
    to_d_link->outer = v_d_link;

    std::vector<size_t> updateMe = { u_v_edge->pmatrix_index, c_v_edge->pmatrix_index,
            v_d_edge->pmatrix_index, a_u_edge->pmatrix_index, u_b_edge->pmatrix_index };
    invalidate_pmatrices(ann_network, updateMe);

    fixReticulations(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.a_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.b_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.c_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.d_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, u, false, visited);
    invalidateHigherCLVs(ann_network, v, false, visited);
    invalidatePmatrixIndex(ann_network, u_b_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, v_d_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, a_u_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, c_v_edge->pmatrix_index, visited);
    invalidatePmatrixIndex(ann_network, u_v_edge->pmatrix_index, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkSanity(network);
    assertReticulationProbs(ann_network);
    invalidateLostMegablobRoots(ann_network, previous_megablob_roots);
}

void updateMoveClvIndex(ArcRemovalMove& move, size_t old_clv_index, size_t new_clv_index) {
    if (move.a_clv_index == old_clv_index) {
        move.a_clv_index = new_clv_index;    
    }
    if (move.b_clv_index == old_clv_index) {
        move.b_clv_index = new_clv_index;
    }
    if (move.c_clv_index == old_clv_index) {
        move.c_clv_index = new_clv_index;
    }
    if (move.d_clv_index == old_clv_index) {
        move.d_clv_index = new_clv_index;
    }
    if (move.u_clv_index == old_clv_index) {
        move.u_clv_index = new_clv_index;
    }
    if (move.v_clv_index == old_clv_index) {
        move.v_clv_index = new_clv_index;
    }
}


void updateMovePmatrixIndex(ArcRemovalMove& move, size_t old_pmatrix_index, size_t new_pmatrix_index) {
    if (move.au_pmatrix_index == old_pmatrix_index) {
        move.au_pmatrix_index = new_pmatrix_index;    
    }
    if (move.cv_pmatric_index == old_pmatrix_index) {
        move.cv_pmatric_index = new_pmatrix_index;
    }
    if (move.ub_pmatrix_index == old_pmatrix_index) {
        move.ub_pmatrix_index = new_pmatrix_index;
    }
    if (move.uv_pmatrix_index == old_pmatrix_index) {
        move.uv_pmatrix_index = new_pmatrix_index;
    }
    if (move.vd_pmatrix_index == old_pmatrix_index) {
        move.vd_pmatrix_index = new_pmatrix_index;
    }
}

void repairConsecutiveClvIndices(AnnotatedNetwork &ann_network, ArcRemovalMove& move) {
    std::unordered_set<size_t> move_clv_indices = {move.a_clv_index, move.b_clv_index, move.c_clv_index, move.d_clv_index, move.u_clv_index, move.v_clv_index};
    std::vector<size_t> missing_clv_indices;
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        if (!ann_network.network.nodes_by_index[i]) {
            missing_clv_indices.emplace_back(i);
            invalidateSingleClv(ann_network.fake_treeinfo, i);
        }
    }

    if (missing_clv_indices.empty()) {
        return;
    }

    for (size_t i = 0; i < ann_network.network.nodes.size(); ++i) {
        if (ann_network.network.nodes[i].clv_index >= ann_network.network.num_nodes() && ann_network.network.nodes[i].clv_index < std::numeric_limits<size_t>::max()) {
            size_t old_clv_index = ann_network.network.nodes[i].clv_index;
            size_t new_clv_index = missing_clv_indices.back();
            // invalidate the clv entry
            invalidateSingleClv(ann_network.fake_treeinfo, old_clv_index);
            if (move_clv_indices.find(old_clv_index) != move_clv_indices.end()) {
                updateMoveClvIndex(move, old_clv_index, new_clv_index);
            }

            // update all references to this clv index
            ann_network.network.nodes[i].clv_index = new_clv_index;
            ann_network.network.nodes_by_index[new_clv_index] = &ann_network.network.nodes[i];
            ann_network.network.nodes_by_index[old_clv_index] = nullptr;
            for (size_t j = 0; j < ann_network.network.nodes[i].links.size(); ++j) {
                ann_network.network.nodes[i].links[j].node_clv_index = new_clv_index;
            }

            missing_clv_indices.pop_back();
        }
    }
}

void repairConsecutivePmatrixIndices(AnnotatedNetwork &ann_network, ArcRemovalMove& move) {
    std::unordered_set<size_t> move_pmatrix_indices = {move.au_pmatrix_index, move.cv_pmatric_index, move.ub_pmatrix_index, move.uv_pmatrix_index, move.vd_pmatrix_index};
    std::vector<size_t> missing_pmatrix_indices;
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        if (!ann_network.network.edges_by_index[i]) {
            missing_pmatrix_indices.emplace_back(i);
            std::vector<bool> visited(ann_network.network.edges.size(), false);
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                ann_network.fake_treeinfo->pmatrix_valid[p][i] = 0;
            }
        }
    }

    if (missing_pmatrix_indices.empty()) {
        return;
    }

    for (size_t i = 0; i < ann_network.network.edges.size(); ++i) {
        if (ann_network.network.edges[i].pmatrix_index >= ann_network.network.num_branches() && ann_network.network.edges[i].pmatrix_index < std::numeric_limits<size_t>::max()) {
            size_t old_pmatrix_index = ann_network.network.edges[i].pmatrix_index;
            size_t new_pmatrix_index = missing_pmatrix_indices.back();
            // invalidate the pmatrix entry
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                ann_network.fake_treeinfo->pmatrix_valid[p][old_pmatrix_index] = 0;
            }
            if (move_pmatrix_indices.find(old_pmatrix_index) != move_pmatrix_indices.end()) {
                updateMovePmatrixIndex(move, old_pmatrix_index, new_pmatrix_index);
            }

            // update all references to this pmatrix index
            ann_network.network.edges[i].pmatrix_index = new_pmatrix_index;
            ann_network.network.edges_by_index[new_pmatrix_index] = &ann_network.network.edges[i];
            ann_network.network.edges_by_index[old_pmatrix_index] = nullptr;

            ann_network.network.edges[i].link1->edge_pmatrix_index = new_pmatrix_index;
            ann_network.network.edges[i].link2->edge_pmatrix_index = new_pmatrix_index;

            missing_pmatrix_indices.pop_back();
        }
    }
}

void repairConsecutiveIndices(AnnotatedNetwork &ann_network, ArcRemovalMove& move) {
    // ensure that pmatrix indices and clv indices remain consecutive. Do the neccessary relabelings.
    repairConsecutiveClvIndices(ann_network, move);
    repairConsecutivePmatrixIndices(ann_network, move);
}

void performMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move) {
    Network &network = ann_network.network;
    std::vector<Node*> previous_megablob_roots = ann_network.blobInfo.megablob_roots;
    Link *from_a_link = getLinkToNode(network, move.a_clv_index, move.u_clv_index);
    Link *to_b_link = getLinkToNode(network, move.b_clv_index, move.u_clv_index);
    Link *from_c_link = getLinkToNode(network, move.c_clv_index, move.v_clv_index);
    Link *to_d_link = getLinkToNode(network, move.d_clv_index, move.v_clv_index);

    std::vector<double> a_b_edge_length = move.a_b_len;
    std::vector<double> c_d_edge_length = move.c_d_len;

    size_t a_u_edge_index = getEdgeTo(network, move.a_clv_index, move.u_clv_index)->pmatrix_index;
    size_t u_b_edge_index = getEdgeTo(network, move.u_clv_index, move.b_clv_index)->pmatrix_index;
    size_t c_v_edge_index = getEdgeTo(network, move.c_clv_index, move.v_clv_index)->pmatrix_index;
    size_t v_d_edge_index = getEdgeTo(network, move.v_clv_index, move.d_clv_index)->pmatrix_index;
    size_t u_v_edge_index = getEdgeTo(network, move.u_clv_index, move.v_clv_index)->pmatrix_index;

    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
    }
    removeNode(network, network.nodes_by_index[move.u_clv_index]);
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
    }
    removeNode(network, network.nodes_by_index[move.v_clv_index]);
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
    }
    removeEdge(network, network.edges_by_index[a_u_edge_index]);
    removeEdge(network, network.edges_by_index[u_b_edge_index]);
    removeEdge(network, network.edges_by_index[c_v_edge_index]);
    removeEdge(network, network.edges_by_index[v_d_edge_index]);
    removeEdge(network, network.edges_by_index[u_v_edge_index]);

    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
    }

    Edge *a_b_edge = addEdge(ann_network, from_a_link, to_b_link, a_b_edge_length,
            move.wanted_ab_pmatrix_index); // was ub before
    Edge *c_d_edge = addEdge(ann_network, from_c_link, to_d_link, c_d_edge_length,
            move.wanted_cd_pmatrix_index); // was vd before

    repairConsecutiveIndices(ann_network, move);

    Node *b = network.nodes_by_index[move.b_clv_index];
    if (b->type == NodeType::RETICULATION_NODE) {
        // u is no longer parent of b, but a is now the parent
        Link *badToParentLink = nullptr;
        if (getReticulationFirstParentPmatrixIndex(b) == u_b_edge_index) {
            badToParentLink = b->getReticulationData()->link_to_first_parent;
        } else {
            assert(getReticulationSecondParentPmatrixIndex(b) == u_b_edge_index);
            badToParentLink = b->getReticulationData()->link_to_second_parent;
        }
        badToParentLink->outer = from_a_link;
        badToParentLink->outer->outer = badToParentLink;
    }

    Node *d = network.nodes_by_index[move.d_clv_index];
    if (d->type == NodeType::RETICULATION_NODE) {
        // v is no longer parent of d, but c is now the parent
        Link *badToParentLink = nullptr;
        if (getReticulationFirstParentPmatrixIndex(d) == v_d_edge_index) {
            badToParentLink = d->getReticulationData()->link_to_first_parent;
        } else {
            assert(getReticulationSecondParentPmatrixIndex(d) == v_d_edge_index);
            badToParentLink = d->getReticulationData()->link_to_second_parent;
        }
        badToParentLink->outer = from_c_link;
        badToParentLink->outer->outer = badToParentLink;
    }
    std::vector<size_t> updateMe;
    if (c_d_edge) {
        updateMe = { a_b_edge->pmatrix_index, c_d_edge->pmatrix_index };
    } else {
        updateMe = { a_b_edge->pmatrix_index };
    }
    invalidate_pmatrices(ann_network, updateMe);

    from_a_link->outer = to_b_link;
    to_b_link->outer = from_a_link;
    from_c_link->outer = to_d_link;
    to_d_link->outer = from_c_link;
    from_a_link->edge_pmatrix_index = a_b_edge->pmatrix_index;
    to_b_link->edge_pmatrix_index = a_b_edge->pmatrix_index;
    from_c_link->edge_pmatrix_index = c_d_edge->pmatrix_index;
    to_d_link->edge_pmatrix_index = c_d_edge->pmatrix_index;

    fixReticulations(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.a_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.b_clv_index], false, visited);
    invalidatePmatrixIndex(ann_network, a_b_edge->pmatrix_index, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.c_clv_index], false, visited);
    invalidateHigherCLVs(ann_network, network.nodes_by_index[move.d_clv_index], false, visited);
    invalidatePmatrixIndex(ann_network, c_d_edge->pmatrix_index, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkSanity(network);
    assertReticulationProbs(ann_network);
    invalidateLostMegablobRoots(ann_network, previous_megablob_roots);
}

void undoMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move) {
    Network &network = ann_network.network;
    Node *a = network.nodes_by_index[move.a_clv_index];
    Node *b = network.nodes_by_index[move.b_clv_index];
    Node *c = network.nodes_by_index[move.c_clv_index];
    Node *d = network.nodes_by_index[move.d_clv_index];

    Node *u = nullptr;
    Node *v = nullptr;
// Find u and v
    std::vector<Node*> uCandidates = getChildren(network, a);
    std::vector<Node*> vCandidates = getChildren(network, c);
    for (size_t i = 0; i < uCandidates.size(); ++i) {
        if (!hasChild(network, uCandidates[i], b)) {
            continue;
        }
        for (size_t j = 0; j < vCandidates.size(); ++j) {
            if (hasChild(network, uCandidates[i], b)
                    && hasChild(network, uCandidates[i], vCandidates[j])
                    && hasChild(network, vCandidates[j], d)) {
                Node *u_cand = uCandidates[i];
                Node *v_cand = vCandidates[j];
                if (u_cand != a && u_cand != b && u_cand != c && u_cand != d && v_cand != a
                        && v_cand != b && v_cand != c && v_cand != d && u_cand != v_cand) {
                    u = u_cand;
                    v = v_cand;
                    break;
                }
            }
        }
        if (u != nullptr && v != nullptr) {
            break;
        }
    }
    assert(u);
    assert(v);
    ArcRemovalMove removal = buildArcRemovalMove(move.a_clv_index, move.b_clv_index,
            move.c_clv_index, move.d_clv_index, u->clv_index, v->clv_index, move.u_v_len, move.c_v_len,
            move.a_u_len, move.a_b_len, move.c_d_len, move.v_d_len, move.u_b_len, MoveType::ArcRemovalMove);
    removal.wanted_ab_pmatrix_index = move.ab_pmatrix_index;
    removal.wanted_cd_pmatrix_index = move.cd_pmatrix_index;
    performMove(ann_network, removal);
}

void undoMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move) {
    ArcInsertionMove insertion = buildArcInsertionMove(move.a_clv_index, move.b_clv_index,
            move.c_clv_index, move.d_clv_index, move.u_v_len, move.c_v_len, move.a_u_len, move.a_b_len, move.c_d_len, move.v_d_len, move.u_b_len, MoveType::ArcInsertionMove);

    // TODO: this likely doesn't work this way now that arc removal moves ensure consecutive indices. Those wanted indices might be already in use...
    insertion.wanted_u_clv_index = move.u_clv_index;
    insertion.wanted_v_clv_index = move.v_clv_index;
    insertion.wanted_au_pmatrix_index = move.au_pmatrix_index;
    insertion.wanted_ub_pmatrix_index = move.ub_pmatrix_index;
    insertion.wanted_cv_pmatrix_index = move.cv_pmatric_index;
    insertion.wanted_vd_pmatrix_index = move.vd_pmatrix_index;
    insertion.wanted_uv_pmatrix_index = move.uv_pmatrix_index;

    performMove(ann_network, insertion);
}

std::string toString(RNNIMove &move) {
    std::stringstream ss;
    std::unordered_map<RNNIMoveType, std::string> lookup;
    lookup[RNNIMoveType::ONE] = "ONE";
    lookup[RNNIMoveType::ONE_STAR] = "ONE_STAR";
    lookup[RNNIMoveType::TWO] = "TWO";
    lookup[RNNIMoveType::TWO_STAR] = "TWO_STAR";
    lookup[RNNIMoveType::THREE] = "THREE";
    lookup[RNNIMoveType::THREE_STAR] = "THREE_STAR";
    lookup[RNNIMoveType::FOUR] = "FOUR";
    ss << lookup[move.type] << ":\n";
    ss << "  u = " << move.u_clv_index << "\n";
    ss << "  v = " << move.v_clv_index << "\n";
    ss << "  s = " << move.s_clv_index << "\n";
    ss << "  t = " << move.t_clv_index << "\n";
    return ss.str();
}

std::string toString(RSPRMove &move) {
    std::stringstream ss;
    ss << "rSPR move:\n";
    ss << "  x_prime = " << move.x_prime_clv_index << "\n";
    ss << "  y_prime = " << move.y_prime_clv_index << "\n";
    ss << "  x = " << move.x_clv_index << "\n";
    ss << "  y = " << move.y_clv_index << "\n";
    ss << "  z = " << move.z_clv_index << "\n";
    return ss.str();
}

std::string toString(ArcInsertionMove &move) {
    std::stringstream ss;
    ss << "arc insertion move:\n";
    ss << "  a = " << move.a_clv_index << "\n";
    ss << "  b = " << move.b_clv_index << "\n";
    ss << "  c = " << move.c_clv_index << "\n";
    ss << "  d = " << move.d_clv_index << "\n";
    return ss.str();

}

std::string toString(ArcRemovalMove &move) {
    std::stringstream ss;
    ss << "arc removal move:\n";
    ss << "  a = " << move.a_clv_index << "\n";
    ss << "  b = " << move.b_clv_index << "\n";
    ss << "  c = " << move.c_clv_index << "\n";
    ss << "  d = " << move.d_clv_index << "\n";
    ss << "  u = " << move.u_clv_index << "\n";
    ss << "  v = " << move.v_clv_index << "\n";
    return ss.str();
}

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RNNIMove &move) {
    Node *u = ann_network.network.nodes_by_index[move.u_clv_index];
    Node *v = ann_network.network.nodes_by_index[move.v_clv_index];
    Node *s = ann_network.network.nodes_by_index[move.s_clv_index];
    Node *t = ann_network.network.nodes_by_index[move.t_clv_index];
    Edge *u_v_edge = getEdgeTo(ann_network.network, u, v);
    Edge *v_s_edge = getEdgeTo(ann_network.network, v, s);
    Edge *u_t_edge = getEdgeTo(ann_network.network, u, t);
    return {u_v_edge->pmatrix_index, v_s_edge->pmatrix_index, u_t_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RNNIMove &move) {
    Node *u = ann_network.network.nodes_by_index[move.u_clv_index];
    Node *v = ann_network.network.nodes_by_index[move.v_clv_index];
    Node *s = ann_network.network.nodes_by_index[move.s_clv_index];
    Node *t = ann_network.network.nodes_by_index[move.t_clv_index];
    Edge *u_s_edge = getEdgeTo(ann_network.network, u, s);
    Edge *v_t_edge = getEdgeTo(ann_network.network, v, t);
    Edge *u_t_edge = getEdgeTo(ann_network.network, u, t);
    return {u_s_edge->pmatrix_index, v_t_edge->pmatrix_index, u_t_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RSPRMove &move) {
    Node *x = ann_network.network.nodes_by_index[move.x_clv_index];
    Node *y = ann_network.network.nodes_by_index[move.y_clv_index];
    Node *x_prime = ann_network.network.nodes_by_index[move.x_prime_clv_index];
    Node *y_prime = ann_network.network.nodes_by_index[move.y_prime_clv_index];
    Node *z = ann_network.network.nodes_by_index[move.z_clv_index];
    Edge *x_y_edge = getEdgeTo(ann_network.network, x, y);
    Edge *x_prime_z_edge = getEdgeTo(ann_network.network, x_prime, z);
    Edge *z_y_prime_edge = getEdgeTo(ann_network.network, z, y_prime);
    return {x_y_edge->pmatrix_index, x_prime_z_edge->pmatrix_index, z_y_prime_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RSPRMove &move) {
    Node *x = ann_network.network.nodes_by_index[move.x_clv_index];
    Node *y = ann_network.network.nodes_by_index[move.y_clv_index];
    Node *x_prime = ann_network.network.nodes_by_index[move.x_prime_clv_index];
    Node *y_prime = ann_network.network.nodes_by_index[move.y_prime_clv_index];
    Node *z = ann_network.network.nodes_by_index[move.z_clv_index];
    Edge *x_prime_y_prime_edge = getEdgeTo(ann_network.network, x_prime, y_prime);
    Edge *x_z_edge = getEdgeTo(ann_network.network, x, z);
    Edge *z_y_edge = getEdgeTo(ann_network.network, z, y);
    return {x_prime_y_prime_edge->pmatrix_index, x_z_edge->pmatrix_index, z_y_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move) {
    Network &network = ann_network.network;
    Node *a = network.nodes_by_index[move.a_clv_index];
    Node *b = network.nodes_by_index[move.b_clv_index];
    Node *c = network.nodes_by_index[move.c_clv_index];
    Node *d = network.nodes_by_index[move.d_clv_index];

    // find u and v
    Node *u = nullptr;
    Node *v = nullptr;
    std::vector<Node*> uCandidates = getChildren(network, a);
    std::vector<Node*> vCandidates = getChildren(network, c);
    for (size_t i = 0; i < uCandidates.size(); ++i) {
        if (!hasChild(network, uCandidates[i], b)) {
            continue;
        }
        for (size_t j = 0; j < vCandidates.size(); ++j) {
            if (hasChild(network, uCandidates[i], b)
                    && hasChild(network, uCandidates[i], vCandidates[j])
                    && hasChild(network, vCandidates[j], d)) {
                Node *u_cand = uCandidates[i];
                Node *v_cand = vCandidates[j];
                if (u_cand != a && u_cand != b && u_cand != c && u_cand != d && v_cand != a
                        && v_cand != b && v_cand != c && v_cand != d && u_cand != v_cand) {
                    u = u_cand;
                    v = v_cand;
                    break;
                }
            }
        }
        if (u != nullptr && v != nullptr) {
            break;
        }
    }
    assert(u);
    assert(v);

    Edge *a_u_edge = getEdgeTo(network, a, u);
    Edge *u_b_edge = getEdgeTo(network, u, b);
    Edge *c_v_edge = getEdgeTo(network, c, v);
    Edge *v_d_edge = getEdgeTo(network, v, d);
    Edge *u_v_edge = getEdgeTo(network, u, v);
    return {a_u_edge->pmatrix_index, u_b_edge->pmatrix_index,c_v_edge->pmatrix_index,v_d_edge->pmatrix_index,u_v_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move) {
    Node *a = ann_network.network.nodes_by_index[move.a_clv_index];
    Node *b = ann_network.network.nodes_by_index[move.b_clv_index];
    Node *c = ann_network.network.nodes_by_index[move.c_clv_index];
    Node *d = ann_network.network.nodes_by_index[move.d_clv_index];
    Edge *a_b_edge = getEdgeTo(ann_network.network, a, b);
    Edge *c_d_edge = getEdgeTo(ann_network.network, c, d);
    return {a_b_edge->pmatrix_index, c_d_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, ArcRemovalMove &move) {
    Node *a = ann_network.network.nodes_by_index[move.a_clv_index];
    Node *b = ann_network.network.nodes_by_index[move.b_clv_index];
    Node *c = ann_network.network.nodes_by_index[move.c_clv_index];
    Node *d = ann_network.network.nodes_by_index[move.d_clv_index];
    Edge *a_b_edge = getEdgeTo(ann_network.network, a, b);
    Edge *c_d_edge = getEdgeTo(ann_network.network, c, d);
    return {a_b_edge->pmatrix_index, c_d_edge->pmatrix_index};
}
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcRemovalMove &move) {
    Node *a = ann_network.network.nodes_by_index[move.a_clv_index];
    Node *b = ann_network.network.nodes_by_index[move.b_clv_index];
    Node *c = ann_network.network.nodes_by_index[move.c_clv_index];
    Node *d = ann_network.network.nodes_by_index[move.d_clv_index];
    Node *u = ann_network.network.nodes_by_index[move.u_clv_index];
    Node *v = ann_network.network.nodes_by_index[move.v_clv_index];
    Edge *a_u_edge = getEdgeTo(ann_network.network, a, u);
    Edge *u_b_edge = getEdgeTo(ann_network.network, u, b);
    Edge *c_v_edge = getEdgeTo(ann_network.network, c, v);
    Edge *v_d_edge = getEdgeTo(ann_network.network, v, d);
    Edge *u_v_edge = getEdgeTo(ann_network.network, u, v);
    return {a_u_edge->pmatrix_index, u_b_edge->pmatrix_index,c_v_edge->pmatrix_index,v_d_edge->pmatrix_index,u_v_edge->pmatrix_index};
}

size_t getRandomIndex(std::mt19937& rng, size_t n) {
    std::uniform_int_distribution<std::mt19937::result_type> d(0, n-1);
    return d(rng);
}

Edge* getRandomEdge(AnnotatedNetwork &ann_network) {
    size_t n = ann_network.network.num_branches();
    return ann_network.network.edges_by_index[getRandomIndex(ann_network.rng, n)];
}

ArcInsertionMove randomArcInsertionMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    std::unordered_set<Edge*> tried;
    while (tried.size() != ann_network.network.num_branches()) {
        Edge* edge = getRandomEdge(ann_network);
        if (tried.count(edge) > 0) {
            continue;
        } else {
            tried.emplace(edge);
        }
        auto moves = possibleArcInsertionMoves(ann_network, edge);
        if (!moves.empty()) {
            return moves[getRandomIndex(ann_network.rng, moves.size())];
        }
    }
    throw std::runtime_error("No random move found");
}

ArcRemovalMove randomArcRemovalMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    auto moves = possibleArcRemovalMoves(ann_network);
    if (!moves.empty()) {
        return moves[getRandomIndex(ann_network.rng, moves.size())];
    } else {
        throw std::runtime_error("No possible move found");
    }
}

ArcInsertionMove randomDeltaPlusMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    std::unordered_set<Edge*> tried;
    while (tried.size() != ann_network.network.num_branches()) {
        Edge* edge = getRandomEdge(ann_network);
        if (tried.count(edge) > 0) {
            continue;
        } else {
            tried.emplace(edge);
        }
        auto moves = possibleDeltaPlusMoves(ann_network, edge);
        if (!moves.empty()) {
            return moves[getRandomIndex(ann_network.rng, moves.size())];
        }
    }
    throw std::runtime_error("No random move found");
}

ArcRemovalMove randomDeltaMinusMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    auto moves = possibleDeltaMinusMoves(ann_network);
    if (!moves.empty()) {
        return moves[getRandomIndex(ann_network.rng, moves.size())];
    } else {
        throw std::runtime_error("No possible move found");
    }
}

RNNIMove randomRNNIMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    std::unordered_set<Edge*> tried;
    while (tried.size() != ann_network.network.num_branches()) {
        Edge* edge = getRandomEdge(ann_network);
        if (tried.count(edge) > 0) {
            continue;
        } else {
            tried.emplace(edge);
        }
        auto moves = possibleRNNIMoves(ann_network, edge);
        if (!moves.empty()) {
            return moves[getRandomIndex(ann_network.rng, moves.size())];
        }
    }
    throw std::runtime_error("No random move found");
}

RSPRMove randomRSPRMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    std::unordered_set<Edge*> tried;
    while (tried.size() != ann_network.network.num_branches()) {
        Edge* edge = getRandomEdge(ann_network);
        if (tried.count(edge) > 0) {
            continue;
        } else {
            tried.emplace(edge);
        }
        auto moves = possibleRSPRMoves(ann_network, edge);
        if (!moves.empty()) {
            return moves[getRandomIndex(ann_network.rng, moves.size())];
        }
    }
    throw std::runtime_error("No random move found");
}

RSPRMove randomRSPR1Move(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    std::unordered_set<Edge*> tried;
    while (tried.size() != ann_network.network.num_branches()) {
        Edge* edge = getRandomEdge(ann_network);
        if (tried.count(edge) > 0) {
            continue;
        } else {
            tried.emplace(edge);
        }
        auto moves = possibleRSPR1Moves(ann_network, edge);
        if (!moves.empty()) {
            return moves[getRandomIndex(ann_network.rng, moves.size())];
        }
    }
    throw std::runtime_error("No random move found");
}

RSPRMove randomTailMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    std::unordered_set<Edge*> tried;
    while (tried.size() != ann_network.network.num_branches()) {
        Edge* edge = getRandomEdge(ann_network);
        if (tried.count(edge) > 0) {
            continue;
        } else {
            tried.emplace(edge);
        }
        auto moves = possibleTailMoves(ann_network, edge);
        if (!moves.empty()) {
            return moves[getRandomIndex(ann_network.rng, moves.size())];
        }
    }
    throw std::runtime_error("No random move found");
}

RSPRMove randomHeadMove(AnnotatedNetwork &ann_network) {
    // TODO: This can be made faster
    std::unordered_set<Edge*> tried;
    while (tried.size() != ann_network.network.num_branches()) {
        Edge* edge = getRandomEdge(ann_network);
        if (tried.count(edge) > 0) {
            continue;
        } else {
            tried.emplace(edge);
        }
        auto moves = possibleHeadMoves(ann_network, edge);
        if (!moves.empty()) {
            return moves[getRandomIndex(ann_network.rng, moves.size())];
        }
    }
    throw std::runtime_error("No random move found");
}

}
