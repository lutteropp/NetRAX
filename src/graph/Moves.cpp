/*
 * Moves.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#include "Moves.hpp"
#include "Network.hpp"
#include "NetworkTopology.hpp"
#include "Direction.hpp"
#include "AnnotatedNetwork.hpp"
#include "BiconnectedComponents.hpp"
#include <vector>
#include <queue>
#include <unordered_set>
#include <sstream>
#include <iostream>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

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

RNNIMove buildRNNIMove(size_t u_clv_index, size_t v_clv_index, size_t s_clv_index, size_t t_clv_index,
        RNNIMoveType type) {
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
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index, RNNIMoveType::ONE));
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
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index, RNNIMoveType::TWO));
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
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index, RNNIMoveType::THREE));
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
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index, RNNIMoveType::FOUR));
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

    ReticulationData retData;
    retData.init(reticulationId, label, active, link_to_first_parent, link_to_second_parent, link_to_child);
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
}

void addRepairCandidates(Network &network, std::unordered_set<Node*> &repair_candidates, Node *node) {
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
}

void performMove(AnnotatedNetwork &ann_network, RNNIMove &move) {
    Network &network = ann_network.network;
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    assertBeforeMove(network, move);
    exchangeEdges(network, u, v, s, t);
    updateLinkDirections(network, move);
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR || move.type == RNNIMoveType::THREE
            || move.type == RNNIMoveType::FOUR) {
        switchReticulations(network, u, v);
    }
    fixReticulations(network, move);
    assertAfterMove(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, u);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, v);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, s);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, t);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkReticulationProbs(ann_network);
}

void undoMove(AnnotatedNetwork &ann_network, RNNIMove &move) {
    Network &network = ann_network.network;
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    assertAfterMove(network, move);
    exchangeEdges(network, u, v, t, s); // note that s and t are exchanged here
    updateLinkDirectionsReverse(network, move);
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR || move.type == RNNIMoveType::THREE
            || move.type == RNNIMoveType::FOUR) {
        switchReticulations(network, u, v);
    }
    fixReticulations(network, move);
    assertBeforeMove(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, u);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, v);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, s);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, t);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkReticulationProbs(ann_network);
}

std::vector<std::pair<Node*, Node*> > getZYChoices(Network &network, Node *x_prime, Node *y_prime, Node *x,
        Node *fixed_y = nullptr, bool returnHead = true, bool returnTail = true) {
    std::vector<std::pair<Node*, Node*> > res;
    auto x_prime_children = getChildren(network, x_prime);
    auto x_children = getChildren(network, x);
    for (Node *z : x_children) {
        if (std::find(x_prime_children.begin(), x_prime_children.end(), z) != x_prime_children.end()) {
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

RSPRMove buildRSPRMove(size_t x_prime_clv_index, size_t y_prime_clv_index, size_t x_clv_index, size_t y_clv_index,
        size_t z_clv_index, MoveType moveType) {
    RSPRMove move = RSPRMove();
    move.x_prime_clv_index = x_prime_clv_index;
    move.y_prime_clv_index = y_prime_clv_index;
    move.x_clv_index = x_clv_index;
    move.y_clv_index = y_clv_index;
    move.z_clv_index = z_clv_index;
    move.moveType = moveType;
    return move;
}

void possibleRSPRMovesInternal(std::vector<RSPRMove> &res, Network &network, Node *x_prime, Node *y_prime, Node *x,
        Node *fixed_y, bool returnHead, bool returnTail, MoveType moveType) {
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
                res.emplace_back(
                        buildRSPRMove(x_prime->clv_index, y_prime->clv_index, x->clv_index, y->clv_index, z->clv_index,
                                moveType));
            }
        } else { // tail-moving rSPR move
            if (!hasPath(network, w, x_prime)) {
                res.emplace_back(
                        buildRSPRMove(x_prime->clv_index, y_prime->clv_index, x->clv_index, y->clv_index, z->clv_index,
                                moveType));
            }
        }
    }
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge, Node *fixed_x, Node *fixed_y,
        MoveType moveType, bool returnHead = true, bool returnTail = true) {
    Network &network = ann_network.network;
    std::vector<RSPRMove> res;
    Node *x_prime = getSource(network, edge);
    Node *y_prime = getTarget(network, edge);

    if (fixed_x) {
        possibleRSPRMovesInternal(res, network, x_prime, y_prime, fixed_x, fixed_y, returnHead, returnTail, moveType);
    } else {
        for (size_t i = 0; i < network.num_nodes(); ++i) {
            Node *x = &network.nodes[i];
            possibleRSPRMovesInternal(res, network, x_prime, y_prime, x, fixed_y, returnHead, returnTail, moveType);
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
    std::vector<RSPRMove> case1 = possibleRSPRMoves(ann_network, edge, y_prime, nullptr, MoveType::RSPR1Move);
    res.insert(std::end(res), std::begin(case1), std::end(case1));

    // Case 2: x_prime == x
    std::vector<RSPRMove> case2 = possibleRSPRMoves(ann_network, edge, x_prime, nullptr, MoveType::RSPR1Move);
    res.insert(std::end(res), std::begin(case2), std::end(case2));

    // Case 3: x_prime == y
    std::vector<RSPRMove> case3 = possibleRSPRMoves(ann_network, edge, nullptr, x_prime, MoveType::RSPR1Move);
    res.insert(std::end(res), std::begin(case3), std::end(case3));

    // Case 4: y_prime == y
    std::vector<RSPRMove> case4 = possibleRSPRMoves(ann_network, edge, nullptr, y_prime, MoveType::RSPR1Move);
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

ArcInsertionMove buildArcInsertionMove(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index, size_t d_clv_index,
        double u_v_len, double c_v_len, double u_v_prob, double c_v_prob, double a_u_len, MoveType moveType) {
    ArcInsertionMove move = ArcInsertionMove();
    move.a_clv_index = a_clv_index;
    move.b_clv_index = b_clv_index;
    move.c_clv_index = c_clv_index;
    move.d_clv_index = d_clv_index;

    move.u_v_len = u_v_len;
    move.c_v_len = c_v_len;
    move.u_v_prob = u_v_prob;
    move.c_v_prob = c_v_prob;
    move.a_u_len = a_u_len;
    move.moveType = moveType;
    return move;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, const Edge *edge, Node *c,
        Node *d, MoveType moveType) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    // choose two distinct arcs ab, cd (with cd not ancestral to ab -> no d-a-path allowed)
    Node *a = getSource(network, edge);
    Node *b = getTarget(network, edge);
    double a_b_len = edge->length;

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
                double c_d_len = network.edges_by_index[c->links[i].edge_pmatrix_index]->length;
                res.emplace_back(
                        buildArcInsertionMove(a->clv_index, b->clv_index, c_cand->clv_index, d_cand->clv_index, 1.0,
                                c_d_len / 2, 0.5, 0.5, a_b_len / 2, moveType));
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
                double c_d_len = network.edges_by_index[d->links[i].edge_pmatrix_index]->length;
                res.emplace_back(
                        buildArcInsertionMove(a->clv_index, b->clv_index, c_cand->clv_index, d_cand->clv_index, 1.0,
                                c_d_len / 2, 0.5, 0.5, a_b_len / 2, moveType));
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
                double c_d_len = network.edges[i].length;
                res.emplace_back(
                        buildArcInsertionMove(a->clv_index, b->clv_index, c_cand->clv_index, d_cand->clv_index, 1.0,
                                c_d_len / 2, 0.5, 0.5, a_b_len / 2, moveType));
            }
        }
    }
    return res;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, const Edge *edge) {
    return possibleArcInsertionMoves(ann_network, edge, nullptr, nullptr, MoveType::ArcInsertionMove);
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network, const Edge *edge) {
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
        std::vector<ArcInsertionMove> moves = possibleArcInsertionMoves(ann_network, &network.edges[i], nullptr,
                nullptr, MoveType::ArcInsertionMove);
        res.insert(std::end(res), std::begin(moves), std::end(moves));
    }
    return res;
}

ArcRemovalMove buildArcRemovalMove(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index, size_t d_clv_index,
        size_t u_clv_index, size_t v_clv_index, double u_v_len, double c_v_len, double u_v_prob, double c_v_prob,
        double a_u_len, MoveType moveType) {
    ArcRemovalMove move = ArcRemovalMove();
    move.a_clv_index = a_clv_index;
    move.b_clv_index = b_clv_index;
    move.c_clv_index = c_clv_index;
    move.d_clv_index = d_clv_index;
    move.u_clv_index = u_clv_index;
    move.v_clv_index = v_clv_index;

    move.u_v_len = u_v_len;
    move.c_v_len = c_v_len;
    move.u_v_prob = u_v_prob;
    move.c_v_prob = c_v_prob;
    move.a_u_len = a_u_len;
    move.moveType = moveType;
    return move;
}

std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network, Node *v, MoveType moveType) {
    // v is a reticulation node, u is one parent of v, c is the other parent of v, a is parent of u, d is child of v, b is other child of u
    std::vector<ArcRemovalMove> res;
    Network &network = ann_network.network;
    assert(v->type == NodeType::RETICULATION_NODE);
    Node *d = getReticulationChild(network, v);
    Node *first_parent = getReticulationFirstParent(network, v);
    Node *second_parent = getReticulationSecondParent(network, v);
    std::vector<std::pair<Node*, Node*> > ucChoices;
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
        res.emplace_back(
                buildArcRemovalMove(a->clv_index, b->clv_index, c->clv_index, d->clv_index, u->clv_index, v->clv_index,
                        getEdgeTo(network, u, v)->length, getEdgeTo(network, c, v)->length,
                        getEdgeTo(network, u, v)->prob, getEdgeTo(network, c, v)->prob,
                        getEdgeTo(network, a, u)->length, moveType));
    }
    return res;
}

std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network, Node *v) {
    std::vector<ArcRemovalMove> res;
    std::vector<ArcRemovalMove> allRemovals = possibleArcRemovalMoves(ann_network, v, MoveType::DeltaMinusMove);
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
        auto moves = possibleArcRemovalMoves(ann_network, network.reticulation_nodes[i], MoveType::ArcRemovalMove);
        res.insert(std::end(res), std::begin(moves), std::end(moves));
    }
    return res;
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<ArcInsertionMove> branch_moves = possibleDeltaPlusMoves(ann_network, &network.edges[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network) {
    std::vector<ArcRemovalMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        std::vector<ArcRemovalMove> branch_moves = possibleDeltaMinusMoves(ann_network, network.reticulation_nodes[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    return res;
}

void performMove(AnnotatedNetwork &ann_network, RSPRMove &move) {
    Network &network = ann_network.network;
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

    x_out_link->edge_pmatrix_index = x_y_edge->pmatrix_index;
    y_in_link->edge_pmatrix_index = x_y_edge->pmatrix_index;
    x_prime_out_link->edge_pmatrix_index = x_prime_z_edge->pmatrix_index;
    z_in_link->edge_pmatrix_index = x_prime_z_edge->pmatrix_index;
    z_out_link->edge_pmatrix_index = z_y_prime_edge->pmatrix_index;
    y_prime_in_link->edge_pmatrix_index = z_y_prime_edge->pmatrix_index;

    fixReticulations(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, z);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, x);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, x_prime);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkReticulationProbs(ann_network);
}

void undoMove(AnnotatedNetwork &ann_network, RSPRMove &move) {
    Network &network = ann_network.network;
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

    x_prime_out_link->edge_pmatrix_index = x_prime_y_prime_edge->pmatrix_index;
    y_prime_in_link->edge_pmatrix_index = x_prime_y_prime_edge->pmatrix_index;
    x_out_link->edge_pmatrix_index = x_z_edge->pmatrix_index;
    z_in_link->edge_pmatrix_index = x_z_edge->pmatrix_index;
    z_out_link->edge_pmatrix_index = z_y_edge->pmatrix_index;
    y_in_link->edge_pmatrix_index = z_y_edge->pmatrix_index;

    fixReticulations(network, move);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, z);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, x);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, x_prime);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkReticulationProbs(ann_network);
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

Edge* addEdge(Network &network, Link *link1, Link *link2, double length, double prob, size_t pmatrix_index) {
    assert(network.num_branches() < network.edges.size());
    if (link1->direction == Direction::INCOMING) {
        std::swap(link1, link2);
    }

    if (network.nodes_by_index[link1->node_clv_index]->isTip()
            || network.nodes_by_index[link2->node_clv_index]->isTip()) {
        assert(pmatrix_index < network.num_tips());
    }
    assert(network.edges_by_index[pmatrix_index] == nullptr);
    network.edges[network.branchCount].init(pmatrix_index, link1, link2, length, prob);
    network.edges_by_index[pmatrix_index] = &network.edges[network.branchCount];
    network.branchCount++;

    return network.edges_by_index[pmatrix_index];
}

Edge* addEdge(Network &network, Link *link1, Link *link2, double length, double prob) {
    size_t pmatrix_index = network.edges.size() - 1;
    if (link1->direction == Direction::INCOMING) {
        std::swap(link1, link2);
    }
    if (link1->node_clv_index < network.num_tips()) {
        pmatrix_index = link1->node_clv_index;
    } else if (link2->node_clv_index < network.num_tips()) {
        pmatrix_index = link2->node_clv_index;
    } else {
        // find smallest free non-tip pmatrix index
        for (size_t i = network.num_tips(); i < network.edges.size(); ++i) {
            if (network.edges_by_index[i] == nullptr) {
                pmatrix_index = i;
                break;
            }
        }
    }

    return addEdge(network, link1, link2, length, prob, pmatrix_index);
}

void checkSanity(Network &network) {
    // check pmatrix indices of edges adjacent to tip nodes
    for (size_t i = 0; i < network.num_tips(); ++i) {
        assert(network.nodes_by_index[i]->links[0].edge_pmatrix_index < network.num_tips());
    }

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
        network.reticulation_nodes[node->getReticulationData()->reticulation_index] = &network.nodes[network.nodeCount
                - 1];
    }

    if (network.nodes_by_index[other_index]->type == NodeType::RETICULATION_NODE) {
        network.reticulation_nodes[network.nodes_by_index[other_index]->getReticulationData()->getReticulationIndex()] =
                &network.nodes[index_in_nodes_array];
    }

    if (network.nodes_by_index[other_index]->type == NodeType::RETICULATION_NODE
            && node->type == NodeType::RETICULATION_NODE) {
        unsigned int other_ret_index = network.nodes_by_index[other_index]->getReticulationData()->reticulation_index;
        unsigned int node_ret_index = node->getReticulationData()->reticulation_index;
        if (node_ret_index < other_ret_index) {
            // swap the reticulation indices
            network.reticulation_nodes[other_ret_index]->getReticulationData()->reticulation_index = node_ret_index;
            network.reticulation_nodes[node_ret_index]->getReticulationData()->reticulation_index = other_ret_index;
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

Node* addInnerNode(Network &network, ReticulationData *retData = nullptr) {
    assert(network.num_nodes() < network.nodes.size());
    unsigned int clv_index = network.nodes.size() - 1;
    // try to find a smaller unused clv index
    for (size_t i = 0; i < clv_index; ++i) {
        if (network.nodes_by_index[i] == nullptr) {
            clv_index = i;
            break;
        }
    }
    assert(network.nodes_by_index[clv_index] == nullptr);
    unsigned int scaler_index = clv_index - network.num_tips();
    network.nodes_by_index[clv_index] = &network.nodes[network.nodeCount];

    if (retData) {
        network.nodes[network.nodeCount].initReticulation(clv_index, scaler_index, "", *retData);
        network.reticulation_nodes.emplace_back(network.nodes_by_index[clv_index]);
        network.nodes[network.nodeCount].getReticulationData()->reticulation_index = network.reticulation_nodes.size()
                - 1;
        for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
            assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
        }
    } else {
        network.nodes[network.nodeCount].initBasic(clv_index, scaler_index, "");
    }

    network.nodeCount++;
    return network.nodes_by_index[clv_index];
}

void reloadBranchLengthsAndBranchProbs(AnnotatedNetwork &ann_network, std::vector<size_t> &affectedPmatrixIndices) {
    Network &network = ann_network.network;
    pllmod_treeinfo_t *fake_treeinfo = ann_network.fake_treeinfo;
    unsigned int partitions = 1;
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) { // each partition has its branch length
        partitions = fake_treeinfo->partition_count;
    }
    for (size_t pmatrix_index : affectedPmatrixIndices) {
        assert(network.edges_by_index[pmatrix_index]);
        for (size_t p = 0; p < partitions; ++p) {
            ann_network.branch_probs[p][pmatrix_index] = network.edges_by_index[pmatrix_index]->prob;
            fake_treeinfo->branch_lengths[p][pmatrix_index] = network.edges_by_index[pmatrix_index]->prob;
            fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
        }
    }
    pllmod_treeinfo_update_prob_matrices(fake_treeinfo, 0);
}

void performMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move) {
    Network &network = ann_network.network;

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

    Node *u = addInnerNode(network, nullptr);
    ReticulationData retData;
    retData.init(network.num_reticulations(), "", 0, nullptr, nullptr, nullptr);
    Node *v = addInnerNode(network, &retData);

    Link *to_u_link = make_link(u, nullptr, Direction::INCOMING);
    Link *u_b_link = make_link(u, nullptr, Direction::OUTGOING);
    Link *u_v_link = make_link(u, nullptr, Direction::OUTGOING);

    Link *v_u_link = make_link(v, nullptr, Direction::INCOMING);
    Link *v_c_link = make_link(v, nullptr, Direction::INCOMING);
    Link *v_d_link = make_link(v, nullptr, Direction::OUTGOING);

    double u_v_edge_length = move.u_v_len;
    double c_v_edge_length = move.c_v_len;
    double v_d_edge_length = c_d_edge->length - c_v_edge_length;
    double a_u_edge_length = move.a_u_len;
    double u_b_edge_length = a_b_edge->length - a_u_edge_length;
    double u_v_edge_prob = move.u_v_prob;
    double c_v_edge_prob = move.c_v_prob;
    double v_d_edge_prob = c_d_edge->prob;
    double a_u_edge_prob = 1.0;
    double u_b_edge_prob = a_b_edge->prob;

    removeEdge(network, network.edges_by_index[a_b_edge_index]);
    removeEdge(network, network.edges_by_index[c_d_edge_index]);
    Edge *u_b_edge = addEdge(network, u_b_link, to_b_link, u_b_edge_length, u_b_edge_prob, a_b_edge_index);
    Edge *v_d_edge = addEdge(network, v_d_link, to_d_link, v_d_edge_length, v_d_edge_prob, c_d_edge_index);
    Edge *a_u_edge = addEdge(network, from_a_link, to_u_link, a_u_edge_length, a_u_edge_prob);
    Edge *c_v_edge = addEdge(network, from_c_link, v_c_link, c_v_edge_length, c_v_edge_prob);
    Edge *u_v_edge = addEdge(network, u_v_link, v_u_link, u_v_edge_length, u_v_edge_prob);

    v->getReticulationData()->link_to_first_parent = v_u_link;
    v->getReticulationData()->link_to_second_parent = v_c_link;
    v->getReticulationData()->link_to_child = v_d_link;

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

    std::vector<size_t> updateMe = { c_d_edge_index, a_b_edge_index, u_v_edge->pmatrix_index, c_v_edge->pmatrix_index,
            v_d_edge->pmatrix_index, a_u_edge->pmatrix_index, u_b_edge->pmatrix_index };
    reloadBranchLengthsAndBranchProbs(ann_network, updateMe);

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, network.nodes_by_index[move.d_clv_index]);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, network.nodes_by_index[move.b_clv_index]);
    invalidatePmatrixIndex(ann_network, u_b_edge->pmatrix_index);
    invalidatePmatrixIndex(ann_network, v_d_edge->pmatrix_index);
    invalidatePmatrixIndex(ann_network, a_u_edge->pmatrix_index);
    invalidatePmatrixIndex(ann_network, c_v_edge->pmatrix_index);
    invalidatePmatrixIndex(ann_network, u_v_edge->pmatrix_index);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkSanity(network);
    checkReticulationProbs(ann_network);
}

void performMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move) {
    Network &network = ann_network.network;
    Link *from_a_link = getLinkToNode(network, move.a_clv_index, move.u_clv_index);
    Link *to_b_link = getLinkToNode(network, move.b_clv_index, move.u_clv_index);
    Link *from_c_link = getLinkToNode(network, move.c_clv_index, move.v_clv_index);
    Link *to_d_link = getLinkToNode(network, move.d_clv_index, move.v_clv_index);

    double a_b_edge_length = network.edges_by_index[from_a_link->edge_pmatrix_index]->length
            + network.edges_by_index[to_b_link->edge_pmatrix_index]->length;
    double a_b_edge_prob = network.edges_by_index[to_b_link->edge_pmatrix_index]->prob;
    double c_d_edge_length = network.edges_by_index[from_c_link->edge_pmatrix_index]->length
            + network.edges_by_index[to_d_link->edge_pmatrix_index]->length;
    double c_d_edge_prob = network.edges_by_index[to_d_link->edge_pmatrix_index]->prob;

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

    Edge *a_b_edge = addEdge(network, from_a_link, to_b_link, a_b_edge_length, a_b_edge_prob, u_b_edge_index);
    Edge *c_d_edge = addEdge(network, from_c_link, to_d_link, c_d_edge_length, c_d_edge_prob, v_d_edge_index);

    Node *b = network.nodes_by_index[move.b_clv_index];
    if (b->type == NodeType::RETICULATION_NODE) {
        // u is no longer parent of b, but a is now the parent
        Link *badToParentLink = nullptr;
        if (getReticulationFirstParentPmatrixIndex(network, b) == u_b_edge_index) {
            badToParentLink = b->getReticulationData()->link_to_first_parent;
        } else {
            assert(getReticulationSecondParentPmatrixIndex(network, b) == u_b_edge_index);
            badToParentLink = b->getReticulationData()->link_to_second_parent;
        }
        badToParentLink->outer = from_a_link;
        badToParentLink->outer->outer = badToParentLink;
    }

    Node *d = network.nodes_by_index[move.d_clv_index];
    if (d->type == NodeType::RETICULATION_NODE) {
        // v is no longer parent of d, but c is now the parent
        Link *badToParentLink = nullptr;
        if (getReticulationFirstParentPmatrixIndex(network, d) == v_d_edge_index) {
            badToParentLink = d->getReticulationData()->link_to_first_parent;
        } else {
            assert(getReticulationSecondParentPmatrixIndex(network, d) == v_d_edge_index);
            badToParentLink = d->getReticulationData()->link_to_second_parent;
        }
        badToParentLink->outer = from_c_link;
        badToParentLink->outer->outer = badToParentLink;
    }

    std::vector<size_t> updateMe = { a_b_edge->pmatrix_index, c_d_edge->pmatrix_index };
    reloadBranchLengthsAndBranchProbs(ann_network, updateMe);

    from_a_link->outer = to_b_link;
    to_b_link->outer = from_a_link;
    from_c_link->outer = to_d_link;
    to_d_link->outer = from_c_link;
    from_a_link->edge_pmatrix_index = a_b_edge->pmatrix_index;
    to_b_link->edge_pmatrix_index = a_b_edge->pmatrix_index;
    from_c_link->edge_pmatrix_index = c_d_edge->pmatrix_index;
    to_d_link->edge_pmatrix_index = c_d_edge->pmatrix_index;

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, network.nodes_by_index[move.d_clv_index]);
    invalidateHigherClvs(network, ann_network.fake_treeinfo, visited, network.nodes_by_index[move.b_clv_index]);
    invalidatePmatrixIndex(ann_network, a_b_edge->pmatrix_index);
    invalidatePmatrixIndex(ann_network, c_d_edge->pmatrix_index);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    ann_network.blobInfo = partitionNetworkIntoBlobs(network, ann_network.travbuffer);
    checkSanity(network);
    checkReticulationProbs(ann_network);
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
            if (hasChild(network, uCandidates[i], b) && hasChild(network, uCandidates[i], vCandidates[j])
                    && hasChild(network, vCandidates[j], d)) {
                Node *u_cand = uCandidates[i];
                Node *v_cand = vCandidates[j];
                if (u_cand != a && u_cand != b && u_cand != c && u_cand != d && v_cand != a && v_cand != b
                        && v_cand != c && v_cand != d && u_cand != v_cand) {
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
    ArcRemovalMove removal = buildArcRemovalMove(move.a_clv_index, move.b_clv_index, move.c_clv_index, move.d_clv_index,
            u->clv_index, v->clv_index, getEdgeTo(network, u, v)->length, getEdgeTo(network, c, v)->length,
            getEdgeTo(network, u, v)->prob, getEdgeTo(network, c, v)->prob, getEdgeTo(network, a, u)->length,
            MoveType::ArcRemovalMove);
    performMove(ann_network, removal);
}

void undoMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move) {
    ArcInsertionMove insertion = buildArcInsertionMove(move.a_clv_index, move.b_clv_index, move.c_clv_index,
            move.d_clv_index, move.u_v_len, move.c_v_len, move.u_v_prob, move.c_v_prob, move.a_u_len,
            MoveType::ArcInsertionMove);
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

}
