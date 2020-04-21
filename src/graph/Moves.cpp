/*
 * Moves.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#include "Moves.hpp"
#include "NetworkTopology.hpp"
#include "Direction.hpp"
#include <vector>
#include <queue>
#include <unordered_set>
#include <sstream>

namespace netrax {

bool hasPath(const Network &network, const Node *from, const Node *to, bool nonelementary = false) {
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
        for (const Node *neigh : getAllParents(node)) {
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
std::vector<std::pair<Node*, Node*> > getSTChoices(const Edge &edge) {
    std::vector<std::pair<Node*, Node*> > res;
    Node *u = getSource(edge);
    Node *v = getTarget(edge);

    auto uNeighbors = getNeighbors(u);
    auto vNeighbors = getNeighbors(v);

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

std::vector<RNNIMove> possibleRNNIMoves(Network &network, const Edge &edge) {
    std::vector<RNNIMove> res;
    Node *u = getSource(edge);
    Node *v = getTarget(edge);
    auto stChoices = getSTChoices(edge);
    for (const auto &st : stChoices) {
        Node *s = st.first;
        Node *t = st.second;

        // check for possible variant and add move from the paper if the move would not create a cycle
        if (isOutgoing(u, s) && isOutgoing(v, t)) {
            if (!hasPath(network, s, v)) {
                // add move 1
                res.emplace_back(RNNIMove { u, v, s, t, RNNIMoveType::ONE });
                if (v->type == NodeType::RETICULATION_NODE) {
                    // add move 1*
                    res.emplace_back(RNNIMove { u, v, s, t, RNNIMoveType::ONE_STAR });
                }
            }
        } else if (isOutgoing(s, u) && isOutgoing(t, v)) {
            if (!hasPath(network, u, t)) {
                // add move 2
                res.emplace_back(RNNIMove { u, v, s, t, RNNIMoveType::TWO });
                if (u->type != NodeType::RETICULATION_NODE) {
                    // add move 2*
                    res.emplace_back(RNNIMove { u, v, s, t, RNNIMoveType::TWO_STAR });
                }
            }
        } else if (isOutgoing(s, u) && isOutgoing(v, t)) {
            if (u->type == NodeType::RETICULATION_NODE && v->type != NodeType::RETICULATION_NODE) {
                // add move 3
                res.emplace_back(RNNIMove { u, v, s, t, RNNIMoveType::THREE });
            }
            if (!hasPath(network, u, v, true)) {
                // add move 3*
                res.emplace_back(RNNIMove { u, v, s, t, RNNIMoveType::THREE_STAR });
            }
        } else if (isOutgoing(u, s) && isOutgoing(t, v)) {
            if (!hasPath(network, s, t)) {
                // add move 4
                res.emplace_back(RNNIMove { u, v, s, t, RNNIMoveType::FOUR });
            }
        }
    }
    return res;
}

void exchangeEdges(Node *u, Node *v, Node *s, Node *t) {
    // The edge between {u,s} will now be between {u, t} and the edge between {v,t} will now be between {v,s}. The edge directions stay the same.
    Link *from_u_link = getLinkToNode(u, s);
    Link *from_s_link = getLinkToNode(s, u);
    Link *from_v_link = getLinkToNode(v, t);
    Link *from_t_link = getLinkToNode(t, v);
    Edge *u_s_edge = getEdgeTo(u, s);
    Edge *v_t_edge = getEdgeTo(v, t);

    from_u_link->outer = from_t_link;
    from_t_link->outer = from_u_link;
    from_v_link->outer = from_s_link;
    from_s_link->outer = from_v_link;

    // u_s_edge now becomes u_t edge
    Edge *u_t_edge = u_s_edge;
    u_t_edge->link1 = from_u_link;
    u_t_edge->link2 = from_t_link;
    from_u_link->edge = u_t_edge;
    from_t_link->edge = u_t_edge;

    // v_t_edge now becomes v_s_edge
    Edge *v_s_edge = v_t_edge;
    v_s_edge->link1 = from_v_link;
    v_s_edge->link2 = from_s_link;
    from_v_link->edge = v_s_edge;
    from_s_link->edge = v_s_edge;
}

void changeEdgeDirection(Node *u, Node *v) {
    Link *from_u_link = getLinkToNode(u, v);
    Link *from_v_link = getLinkToNode(v, u);
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

    size_t num_partitions = old_ret_node->reticulationData->prob.size();
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

    ReticulationData retData;
    retData.init(reticulationId, label, active, link_to_first_parent, link_to_second_parent, link_to_child, 0.5,
            num_partitions);
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

void addRepairCandidates(std::unordered_set<Node*> &repair_candidates, Node *node) {
    repair_candidates.emplace(node);
    for (Node *neigh : getNeighbors(node)) {
        repair_candidates.emplace(neigh);
    }
}

void fixReticulations(RNNIMove &move) {
    std::unordered_set<Node*> repair_candidates;
    addRepairCandidates(repair_candidates, move.s);
    addRepairCandidates(repair_candidates, move.t);
    addRepairCandidates(repair_candidates, move.u);
    addRepairCandidates(repair_candidates, move.v);
    for (Node *node : repair_candidates) {
        if (node->type == NodeType::RETICULATION_NODE) {
            resetReticulationLinks(node);
        }
    }
}

void fixReticulations(RSPRMove &move) {
    std::unordered_set<Node*> repair_candidates;
    addRepairCandidates(repair_candidates, move.x);
    addRepairCandidates(repair_candidates, move.x_prime);
    addRepairCandidates(repair_candidates, move.y);
    addRepairCandidates(repair_candidates, move.y_prime);
    addRepairCandidates(repair_candidates, move.z);
    for (Node *node : repair_candidates) {
        if (node->type == NodeType::RETICULATION_NODE) {
            resetReticulationLinks(node);
        }
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

void assertBeforeMove(RNNIMove &move) {
    Node *notReticulation = nullptr;
    Node *reticulation = nullptr;
    if (move.type == RNNIMoveType::ONE_STAR) {
        notReticulation = move.u;
        reticulation = move.v;
    } else if (move.type == RNNIMoveType::TWO_STAR) {
        notReticulation = move.u;
        reticulation = move.v;
    } else if (move.type == RNNIMoveType::THREE) {
        notReticulation = move.v;
        reticulation = move.u;
    }
    checkReticulationProperties(notReticulation, reticulation);
}

void assertAfterMove(RNNIMove &move) {
    Node *notReticulation = nullptr;
    Node *reticulation = nullptr;
    if (move.type == RNNIMoveType::ONE_STAR) {
        notReticulation = move.v;
        reticulation = move.u;
    } else if (move.type == RNNIMoveType::TWO_STAR) {
        notReticulation = move.u;
        reticulation = move.v;
    } else if (move.type == RNNIMoveType::THREE) {
        notReticulation = move.u;
        reticulation = move.v;
    }
    checkReticulationProperties(notReticulation, reticulation);
}

void performMove(Network &network, RNNIMove &move) {
    assertBeforeMove(move);
    exchangeEdges(move.u, move.v, move.s, move.t);
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE_STAR) {
        changeEdgeDirection(move.u, move.v);
    }
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE) {
        switchReticulations(network, move.u, move.v);
    }
    fixReticulations(move);
    assertAfterMove(move);
}

void undoMove(Network &network, RNNIMove &move) {
    exchangeEdges(move.u, move.v, move.t, move.s); // note that s and t are exchanged here
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE_STAR) {
        changeEdgeDirection(move.u, move.v);
    }
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE) {
        switchReticulations(network, move.u, move.v);
    }
    fixReticulations(move);
}

std::vector<std::pair<Node*, Node*> > getZYChoices(Node *x_prime, Node *y_prime, Node *x) {
    std::vector<std::pair<Node*, Node*> > res;
    auto x_prime_children = getChildren(x_prime, getActiveParent(x_prime));
    auto x_children = getChildren(x, getActiveParent(x));
    for (Node *z : x_children) {
        if (std::find(x_prime_children.begin(), x_prime_children.end(), z) != x_prime_children.end()) {
            continue;
        }
        auto z_children = getChildren(z, x);
        if (std::find(z_children.begin(), z_children.end(), y_prime) != z_children.end()) {
            continue;
        }
        for (Node *y : z_children) {
            if (std::find(x_children.begin(), x_children.end(), y) != x_children.end()) {
                continue;
            }
            res.emplace_back(std::make_pair(z, y));
        }
    }
    return res;
}

std::vector<RSPRMove> possibleRSPRMoves(Network &network, const Edge &edge) {
    std::vector<RSPRMove> res;
    Node *x_prime = getSource(edge);
    Node *y_prime = getTarget(edge);

    for (size_t i = 0; i < network.num_nodes(); ++i) {
        Node *x = &network.nodes[i];
        auto zy = getZYChoices(x_prime, y_prime, x);
        for (const auto &entry : zy) {
            Node *z = entry.first;
            Node *y = entry.second;

            Node *w = nullptr;
            auto zNeighbors = getNeighbors(z);
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
                    res.emplace_back(RSPRMove { x_prime, y_prime, x, z, y });
                }
            } else { // tail-moving rSPR move
                if (!hasPath(network, w, x_prime)) {
                    res.emplace_back(RSPRMove { x_prime, y_prime, x, z, y });
                }
            }
        }
    }
    return res;
}

void performMove(Network&, RSPRMove &move) {
    Link *x_out_link = getLinkToNode(move.x, move.z);
    Link *z_in_link = getLinkToNode(move.z, move.x);
    Link *z_out_link = getLinkToNode(move.z, move.y);
    Link *x_prime_out_link = getLinkToNode(move.x_prime, move.y_prime);
    Link *y_prime_in_link = getLinkToNode(move.y_prime, move.x_prime);
    Link *y_in_link = getLinkToNode(move.y, move.z);

    Edge *x_z_edge = getEdgeTo(move.x, move.z);
    Edge *z_y_edge = getEdgeTo(move.z, move.y);
    Edge *x_prime_y_prime_edge = getEdgeTo(move.x_prime, move.y_prime);

    assert(x_prime_out_link->edge == x_prime_y_prime_edge);
    assert(y_prime_in_link->edge == x_prime_y_prime_edge);
    assert(x_out_link->edge == x_z_edge);
    assert(z_in_link->edge == x_z_edge);
    assert(z_out_link->edge == z_y_edge);
    assert(y_in_link->edge == z_y_edge);

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

    assert(x_out_link->edge == x_y_edge);
    assert(y_in_link->edge == x_y_edge);
    assert(x_prime_out_link->edge == x_prime_z_edge);
    assert(z_in_link->edge == x_prime_z_edge);
    assert(z_out_link->edge == z_y_prime_edge);
    assert(y_prime_in_link->edge == z_y_prime_edge);

    fixReticulations(move);
}

void undoMove(Network&, RSPRMove &move) {
    Link *x_out_link = getLinkToNode(move.x, move.y);
    Link *z_in_link = getLinkToNode(move.z, move.x_prime);
    Link *z_out_link = getLinkToNode(move.z, move.y_prime);
    Link *x_prime_out_link = getLinkToNode(move.x_prime, move.z);
    Link *y_prime_in_link = getLinkToNode(move.y_prime, move.z);
    Link *y_in_link = getLinkToNode(move.y, move.x);

    Edge *x_y_edge = getEdgeTo(move.x, move.y);
    Edge *x_prime_z_edge = getEdgeTo(move.x_prime, move.z);
    Edge *z_y_prime_edge = getEdgeTo(move.z, move.y_prime);

    assert(x_out_link->edge == x_y_edge);
    assert(y_in_link->edge == x_y_edge);
    assert(x_prime_out_link->edge == x_prime_z_edge);
    assert(z_in_link->edge == x_prime_z_edge);
    assert(z_out_link->edge == z_y_prime_edge);
    assert(y_prime_in_link->edge == z_y_prime_edge);

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

    assert(x_prime_out_link->edge == x_prime_y_prime_edge);
    assert(y_prime_in_link->edge == x_prime_y_prime_edge);
    assert(x_out_link->edge == x_z_edge);
    assert(z_in_link->edge == x_z_edge);
    assert(z_out_link->edge == z_y_edge);
    assert(y_in_link->edge == z_y_edge);

    fixReticulations(move);
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
    ss << "  u = (" << move.u->label << "," << move.u->clv_index << ")" << "\n";
    ss << "  v = (" << move.v->label << "," << move.v->clv_index << ")" << "\n";
    ss << "  s = (" << move.s->label << "," << move.s->clv_index << ")" << "\n";
    ss << "  t = (" << move.t->label << "," << move.t->clv_index << ")" << "\n";
    return ss.str();
}

std::string toString(RSPRMove &move) {
    std::stringstream ss;
    ss << "rSPR move:\n";
    ss << "  x_prime = (" << move.x_prime->label << "," << move.x_prime->clv_index << ")" << "\n";
    ss << "  y_prime = (" << move.y_prime->label << "," << move.y_prime->clv_index << ")" << "\n";
    ss << "  x = (" << move.x->label << "," << move.x->clv_index << ")" << "\n";
    ss << "  y = (" << move.y->label << "," << move.y->clv_index << ")" << "\n";
    ss << "  z = (" << move.z->label << "," << move.z->clv_index << ")" << "\n";
    return ss.str();
}

}
