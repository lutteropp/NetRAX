#include "RNNIMove.hpp"

#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"

#include <cassert>

namespace netrax {

void setLinkDirections(Network &network, Node *u, Node *v) {
    Link *from_u_link = getLinkToNode(network, u, v);
    Link *from_v_link = getLinkToNode(network, v, u);
    from_u_link->direction = Direction::OUTGOING;
    from_v_link->direction = Direction::INCOMING;
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

bool checkSanity(AnnotatedNetwork& ann_network, RNNIMove& move) {
    bool good = true;
    good &= (move.moveType == MoveType::RNNIMove);

    good &= (ann_network.network.nodes_by_index[move.u_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.v_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.s_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.t_clv_index] != nullptr);

    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.s_clv_index], ann_network.network.nodes_by_index[move.u_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.u_clv_index], ann_network.network.nodes_by_index[move.v_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.v_clv_index], ann_network.network.nodes_by_index[move.t_clv_index]));

    good &= (!hasNeighbor(ann_network.network.nodes_by_index[move.u_clv_index], ann_network.network.nodes_by_index[move.t_clv_index]));
    good &= (!hasNeighbor(ann_network.network.nodes_by_index[move.s_clv_index], ann_network.network.nodes_by_index[move.v_clv_index]));

    return good;
}

bool checkSanity(AnnotatedNetwork& ann_network, std::vector<RNNIMove>& moves) {
    bool sane = true;
    for (size_t i = 0; i < moves.size(); ++i) {
        sane &= checkSanity(ann_network, moves[i]);
    }
    return sane;
}

RNNIMove buildRNNIMove(size_t u_clv_index, size_t v_clv_index, size_t s_clv_index,
        size_t t_clv_index, RNNIMoveType type, size_t edge_orig_idx) {
    RNNIMove move = RNNIMove(edge_orig_idx);
    move.u_clv_index = u_clv_index;
    move.v_clv_index = v_clv_index;
    move.s_clv_index = s_clv_index;
    move.t_clv_index = t_clv_index;
    move.type = type;
    return move;
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

std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network, const Edge *edge) {
    size_t edge_orig_idx = edge->pmatrix_index;
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
                                RNNIMoveType::ONE, edge_orig_idx));
                if (v->type == NodeType::RETICULATION_NODE && u != network.root) {
                    // add move 1*
                    res.emplace_back(
                            buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                    RNNIMoveType::ONE_STAR, edge_orig_idx));
                }
            }
        } else if (isOutgoing(network, s, u) && isOutgoing(network, t, v)) {
            if (!hasPath(network, u, t)) {
                // add move 2
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::TWO, edge_orig_idx));
                if (u->type != NodeType::RETICULATION_NODE) {
                    // add move 2*
                    res.emplace_back(
                            buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                    RNNIMoveType::TWO_STAR, edge_orig_idx));
                }
            }
        } else if (isOutgoing(network, s, u) && isOutgoing(network, v, t)) {
            if (u->type == NodeType::RETICULATION_NODE && v->type != NodeType::RETICULATION_NODE) {
                // add move 3
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::THREE, edge_orig_idx));
            }
            if (!hasPath(network, u, v, true)) {
                // add move 3*
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::THREE_STAR, edge_orig_idx));
            }
        } else if (isOutgoing(network, u, s) && isOutgoing(network, t, v)) {
            if (u != network.root && !hasPath(network, s, t)) {
                // add move 4
                res.emplace_back(
                        buildRNNIMove(u->clv_index, v->clv_index, s->clv_index, t->clv_index,
                                RNNIMoveType::FOUR, edge_orig_idx));
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

bool assertBeforeMove(Network &network, RNNIMove &move) {
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
    return true;
}

bool assertAfterMove(Network &network, RNNIMove &move) {
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
    return true;
}

void performMove(AnnotatedNetwork &ann_network, RNNIMove &move) {
    assert(checkSanity(ann_network, move));
    assert(move.moveType == MoveType::RNNIMove);
    assert(assertConsecutiveIndices(ann_network));
    Network &network = ann_network.network;
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    assert(assertBeforeMove(network, move));
    exchangeEdges(network, u, v, s, t);
    updateLinkDirections(network, move);
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE || move.type == RNNIMoveType::FOUR) {
        switchReticulations(network, u, v);
    }
    fixReticulationLinks(u, v, s, t);
    assert(assertAfterMove(network, move));

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, u, true, visited);
    invalidateHigherCLVs(ann_network, v, true, visited);
    invalidateHigherCLVs(ann_network, s, true, visited);
    invalidateHigherCLVs(ann_network, t, true, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    assert(assertReticulationProbs(ann_network));
    assertConsecutiveIndices(ann_network);
}

void undoMove(AnnotatedNetwork &ann_network, RNNIMove &move) {
    assert(move.moveType == MoveType::RNNIMove);
    assert(assertConsecutiveIndices(ann_network));
    Network &network = ann_network.network;
    Node *u = network.nodes_by_index[move.u_clv_index];
    Node *v = network.nodes_by_index[move.v_clv_index];
    Node *s = network.nodes_by_index[move.s_clv_index];
    Node *t = network.nodes_by_index[move.t_clv_index];
    assert(assertAfterMove(network, move));
    exchangeEdges(network, u, v, t, s); // note that s and t are exchanged here
    updateLinkDirectionsReverse(network, move);
    if (move.type == RNNIMoveType::ONE_STAR || move.type == RNNIMoveType::TWO_STAR
            || move.type == RNNIMoveType::THREE || move.type == RNNIMoveType::FOUR) {
        switchReticulations(network, u, v);
    }
    fixReticulationLinks(u, v, s, t);
    assert(assertBeforeMove(network, move));

    std::vector<bool> visited(network.nodes.size(), false);
    invalidateHigherCLVs(ann_network, u, true, visited);
    invalidateHigherCLVs(ann_network, v, true, visited);
    invalidateHigherCLVs(ann_network, s, true, visited);
    invalidateHigherCLVs(ann_network, t, true, visited);

    ann_network.travbuffer = reversed_topological_sort(ann_network.network);
    assert(assertReticulationProbs(ann_network));
    assert(assertConsecutiveIndices(ann_network));
}

bool isomorphicMoves(const RNNIMove& move1, const RNNIMove& move2) {
    size_t u1 = move1.u_clv_index;
    size_t v1 = move1.v_clv_index;
    size_t s1 = move1.s_clv_index;
    size_t t1 = move1.t_clv_index;
    size_t u2 = move2.u_clv_index;
    size_t v2 = move2.v_clv_index;
    size_t s2 = move2.s_clv_index;
    size_t t2 = move2.t_clv_index;
    return (std::min(u1,v1) == std::min(u2,v2) && std::max(u1,v1) == std::max(u2,v2) && std::min(s1,t1) == std::min(s2,t2) && std::max(s1,t1) == std::max(s2,t2)) 
        || (std::min(u1,v1) == std::min(s2,t2) && std::max(u1,v1) == std::max(s2,t2) && std::min(s1,t1) == std::min(u2,v2) && std::max(s1,t1) == std::max(u2,v2));
}

std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network) {
    std::vector<RNNIMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RNNIMove> branch_moves = possibleRNNIMoves(ann_network, network.edges_by_index[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }

    // filter out duplicates
    size_t cnt = 0;
    for (size_t i = 0; i < res.size(); ++i) {
        bool keep = true;
        for (size_t j = 0; j < i; ++j) {
            if (isomorphicMoves(res[i], res[j])) {
                keep = false;
                break;
            }
        }
        if (keep) {
            res[cnt] = res[i];
            cnt++;
        }
    }
    res.resize(cnt);
    sortByProximity(res, ann_network);
    assert(checkSanity(ann_network, res));
    return res;
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



}