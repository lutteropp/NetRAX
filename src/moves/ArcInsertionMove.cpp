#include "ArcInsertionMove.hpp"
#include "ArcRemovalMove.hpp"

#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"

namespace netrax {

bool checkSanity(AnnotatedNetwork& ann_network, ArcInsertionMove& move) {
    bool good = true;
    good &= (move.moveType == MoveType::ArcInsertionMove || move.moveType == MoveType::DeltaPlusMove);
    good &= (ann_network.network.nodes_by_index[move.a_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.b_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.c_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.d_clv_index] != nullptr);

    good &= (ann_network.network.edges_by_index[move.ab_pmatrix_index] != nullptr);
    good &= (ann_network.network.edges_by_index[move.cd_pmatrix_index] != nullptr);

    good &= (move.a_clv_index != move.b_clv_index);
    good &= (move.c_clv_index != move.d_clv_index);

    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.a_clv_index], ann_network.network.nodes_by_index[move.b_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.c_clv_index], ann_network.network.nodes_by_index[move.d_clv_index]));

    return good;
}

bool checkSanity(AnnotatedNetwork& ann_network, std::vector<ArcInsertionMove>& moves) {
    bool sane = true;
    for (size_t i = 0; i < moves.size(); ++i) {
        sane &= checkSanity(ann_network, moves[i]);
    }
    return sane;
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

ArcInsertionMove buildArcInsertionMove(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
        std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType, size_t edge_orig_idx, size_t node_orig_idx) {
    ArcInsertionMove move = ArcInsertionMove(edge_orig_idx, node_orig_idx);
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
        const Edge *edge, Node *c, Node *d, MoveType moveType, bool noDeltaPlus) {
    size_t edge_orig_idx = edge->pmatrix_index;
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    // choose two distinct arcs ab, cd (with cd not ancestral to ab -> no d-a-path allowed)
    Node *a = getSource(network, edge);
    Node *b = getTarget(network, edge);
    size_t node_orig_idx = b->clv_index;
    std::vector<double> a_b_len = get_edge_lengths(ann_network, edge->pmatrix_index);

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

            if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) || (b->clv_index == c_cand->clv_index) || (b->clv_index == d_cand->clv_index))) {
                continue;
            }

            if (!hasPath(network, d_cand, a)) {
                std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len, u_v_len;

                c_d_len = get_edge_lengths(ann_network, c->links[i].edge_pmatrix_index);
                c_v_len = get_halved_edge_lengths(c_d_len, min_br);
                a_u_len = get_halved_edge_lengths(a_b_len, min_br);
                v_d_len = get_minus_edge_lengths(c_d_len, c_v_len, min_br);
                u_b_len = get_minus_edge_lengths(a_b_len, a_u_len, min_br);
                u_v_len = std::vector<double>(ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1, 1.0);

                ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                        c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                        a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);
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

                if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) || (b->clv_index == c_cand->clv_index) || (b->clv_index == d_cand->clv_index))) {
                    continue;
                }

                std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len, u_v_len;
                c_d_len = get_edge_lengths(ann_network, d->links[i].edge_pmatrix_index);
                c_v_len = get_halved_edge_lengths(c_d_len, min_br);
                a_u_len = get_halved_edge_lengths(a_b_len, min_br);
                v_d_len = get_minus_edge_lengths(c_d_len, c_v_len, min_br);
                u_b_len = get_minus_edge_lengths(a_b_len, a_u_len, min_br);
                u_v_len = std::vector<double>(ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1, 1.0);

                ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                        c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                        a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

                move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                res.emplace_back(move);
            }
        }
    } else {
        for (size_t i = 0; i < network.num_branches(); ++i) {
            if (i == edge->pmatrix_index) {
                continue;
            }
            Node *c_cand = getSource(network, network.edges_by_index[i]);
            Node *d_cand = getTarget(network, network.edges_by_index[i]);

            if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) || (b->clv_index == c_cand->clv_index) || (b->clv_index == d_cand->clv_index))) {
                continue;
            }

            if (!hasPath(network, d_cand, a)) {
                std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len, u_v_len;
                c_d_len = get_edge_lengths(ann_network, i);
                c_v_len = get_halved_edge_lengths(c_d_len, min_br);
                a_u_len = get_halved_edge_lengths(a_b_len, min_br);
                v_d_len = get_minus_edge_lengths(c_d_len, c_v_len, min_br);
                u_b_len = get_minus_edge_lengths(a_b_len, a_u_len, min_br);
                u_v_len = std::vector<double>(ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1, 1.0);

                ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                        c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                        a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

                move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                res.emplace_back(move);
            }
        }
    }
    return res;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        Node *node, Node *c, Node *d, MoveType moveType, bool noDeltaPlus, int min_radius, int max_radius) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    if (node == ann_network.network.root) {
        return res; // because we need a parent
    }

    size_t node_orig_idx = node->clv_index;

    std::vector<Node*> radius_nodes = getNeighborsWithinRadius(network, node, min_radius, max_radius);

    Node *b = node;
    std::vector<Node*> parents;
    if (node->getType() == NodeType::BASIC_NODE) {
        parents.emplace_back(getActiveParent(network, node));
    } else {
        parents.emplace_back(getReticulationFirstParent(network, node));
        parents.emplace_back(getReticulationSecondParent(network, node));
    }
    for (Node* a : parents) {
        Edge* edge = getEdgeTo(network, a, b);
        size_t edge_orig_idx = edge->pmatrix_index;
        // choose two distinct arcs ab, cd (with cd not ancestral to ab -> no d-a-path allowed)
        std::vector<double> a_b_len = get_edge_lengths(ann_network, edge->pmatrix_index);

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
                if (std::find(radius_nodes.begin(), radius_nodes.end(), d_cand) != radius_nodes.end()) {
                    if (a->clv_index == c_cand->clv_index && b->clv_index == d_cand->clv_index) {
                        continue;
                    }

                    if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) || (b->clv_index == c_cand->clv_index) || (b->clv_index == d_cand->clv_index))) {
                        continue;
                    }

                    if (!hasPath(network, d_cand, a)) {
                        std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len, u_v_len;

                        c_d_len = get_edge_lengths(ann_network, c->links[i].edge_pmatrix_index);
                        c_v_len = get_halved_edge_lengths(c_d_len, min_br);
                        a_u_len = get_halved_edge_lengths(a_b_len, min_br);
                        v_d_len = get_minus_edge_lengths(c_d_len, c_v_len, min_br);
                        u_b_len = get_minus_edge_lengths(a_b_len, a_u_len, min_br);
                        u_v_len = std::vector<double>(ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1, min_br);

                        ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                                c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                                a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);
                        move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                        move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                        res.emplace_back(move);
                    }
                }
            }
        } else if (d) {
            d_cand = d;

            if (std::find(radius_nodes.begin(), radius_nodes.end(), d_cand) != radius_nodes.end()) {
                if (!hasPath(network, d_cand, a)) {
                    for (size_t i = 0; i < d->links.size(); ++i) {
                        if (d->links[i].direction == Direction::OUTGOING) {
                            continue;
                        }
                        Node *c_cand = network.nodes_by_index[d->links[i].outer->node_clv_index];
                        if (a->clv_index == c_cand->clv_index && b->clv_index == d_cand->clv_index) {
                            continue;
                        }

                        if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) || (b->clv_index == c_cand->clv_index) || (b->clv_index == d_cand->clv_index))) {
                            continue;
                        }

                        std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len, u_v_len;
                        c_d_len = get_edge_lengths(ann_network, d->links[i].edge_pmatrix_index);
                        c_v_len = get_halved_edge_lengths(c_d_len, min_br);
                        a_u_len = get_halved_edge_lengths(a_b_len, min_br);
                        v_d_len = get_minus_edge_lengths(c_d_len, c_v_len, min_br);
                        u_b_len = get_minus_edge_lengths(a_b_len, a_u_len, min_br);
                        u_v_len = std::vector<double>(ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1, 1.0);

                        ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                                c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                                a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

                        move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                        move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                        res.emplace_back(move);
                    }
                }
            }
        } else {
            std::vector<Node*> d_candidates = radius_nodes;
            for (Node* d_cand : d_candidates) {
                std::vector<Node*> c_candidates = getAllParents(network, d_cand);
                for (Node* c_cand : c_candidates) {
                    Edge* actEdge = getEdgeTo(network, c_cand, d_cand);
                    if (actEdge->pmatrix_index == edge->pmatrix_index) {
                        continue;
                    }
                    if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) || (b->clv_index == c_cand->clv_index) || (b->clv_index == d_cand->clv_index))) {
                        continue;
                    }

                    if (!hasPath(network, d_cand, a)) {
                        std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len, u_v_len;
                        c_d_len = get_edge_lengths(ann_network, actEdge->pmatrix_index);
                        c_v_len = get_halved_edge_lengths(c_d_len, min_br);
                        a_u_len = get_halved_edge_lengths(a_b_len, min_br);
                        v_d_len = get_minus_edge_lengths(c_d_len, c_v_len, min_br);
                        u_b_len = get_minus_edge_lengths(a_b_len, a_u_len, min_br);
                        u_v_len = std::vector<double>(ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1, 1.0);

                        ArcInsertionMove move = buildArcInsertionMove(a->clv_index, b->clv_index,
                                c_cand->clv_index, d_cand->clv_index, u_v_len, c_v_len,
                                a_u_len, a_b_len, c_d_len, v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

                        move.ab_pmatrix_index = getEdgeTo(network, a, b)->pmatrix_index;
                        move.cd_pmatrix_index = getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
                        res.emplace_back(move);
                    }
                }
            }
        }
    }
    return res;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const Edge *edge, bool noDeltaPlus) {
    return possibleArcInsertionMoves(ann_network, edge, nullptr, nullptr,
            MoveType::ArcInsertionMove, noDeltaPlus);
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, Node *node, bool noDeltaPlus, int min_radius, int max_radius) {
    return possibleArcInsertionMoves(ann_network, node, nullptr, nullptr,
            MoveType::ArcInsertionMove, noDeltaPlus, min_radius, max_radius);
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network,
        const Edge *edge) {
    Network &network = ann_network.network;
    std::vector<ArcInsertionMove> res;
    Node *a = getSource(network, edge);
    Node *b = getTarget(network, edge);

// Case 1: a == c
    std::vector<ArcInsertionMove> case1 = possibleArcInsertionMoves(ann_network, edge, a, nullptr,
            MoveType::DeltaPlusMove, false);
    res.insert(std::end(res), std::begin(case1), std::end(case1));

// Case 2: b == d
    std::vector<ArcInsertionMove> case2 = possibleArcInsertionMoves(ann_network, edge, nullptr, b,
            MoveType::DeltaPlusMove, false);
    res.insert(std::end(res), std::begin(case2), std::end(case2));

// Case 3: b == c
    std::vector<ArcInsertionMove> case3 = possibleArcInsertionMoves(ann_network, edge, b, nullptr,
            MoveType::DeltaPlusMove, false);
    res.insert(std::end(res), std::begin(case3), std::end(case3));

    return res;
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network, Node *node, int min_radius, int max_radius) {
    Network &network = ann_network.network;
    std::vector<ArcInsertionMove> res;
    if (node == ann_network.network.root) {
        return res; // because we need a parent, and the root has no parent
    }
    Node *b = node;
    std::vector<Node*> parents = getAllParents(network, node);
    for (Node* a : parents) {
        // Case 1: a == c
        std::vector<ArcInsertionMove> case1 = possibleArcInsertionMoves(ann_network, node, a, nullptr,
                MoveType::DeltaPlusMove, false, min_radius, max_radius);
        res.insert(std::end(res), std::begin(case1), std::end(case1));

        if (min_radius == 0) {
            // Case 2: b == d
            std::vector<ArcInsertionMove> case2 = possibleArcInsertionMoves(ann_network, node, nullptr, b,
                    MoveType::DeltaPlusMove, false, min_radius, max_radius);
            res.insert(std::end(res), std::begin(case2), std::end(case2));
        }

        // Case 3: b == c
        std::vector<ArcInsertionMove> case3 = possibleArcInsertionMoves(ann_network, node, b, nullptr,
                MoveType::DeltaPlusMove, false, min_radius, max_radius);
        res.insert(std::end(res), std::begin(case3), std::end(case3));
    }
    return res;
}


std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, const std::vector<Node*>& start_nodes, bool noDeltaPlus, int min_radius, int max_radius) {
    std::vector<ArcInsertionMove> res;
    for (Node* node : start_nodes) {
        std::vector<ArcInsertionMove> res_node = possibleArcInsertionMoves(ann_network, node, noDeltaPlus, min_radius, max_radius);
        res.insert(std::end(res), std::begin(res_node), std::end(res_node));
    }
    return res;
}

std::vector<ArcInsertionMove> possibleMoves(AnnotatedNetwork& ann_network, const std::vector<Node*>& start_nodes, ArcInsertionMove placeholderMove, int min_radius, int max_radius) {
    return possibleArcInsertionMoves(ann_network, start_nodes, min_radius, max_radius);
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network, const std::vector<Node*>& start_nodes, int min_radius, int max_radius) {
    std::vector<ArcInsertionMove> res;
    for (Node* node : start_nodes) {
        std::vector<ArcInsertionMove> res_node = possibleDeltaPlusMoves(ann_network, node, min_radius, max_radius);
        res.insert(std::end(res), std::begin(res_node), std::end(res_node));
    }
    return res;
}

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, bool noDeltaPlus, int min_radius, int max_radius) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        std::vector<ArcInsertionMove> node_moves = possibleArcInsertionMoves(ann_network,
                network.nodes_by_index[i], nullptr, nullptr, MoveType::ArcInsertionMove, noDeltaPlus, min_radius, max_radius);
        res.insert(std::end(res), std::begin(node_moves), std::end(node_moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanity(ann_network, res));
    return res;
}

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network, int min_radius, int max_radius) {
    std::vector<ArcInsertionMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        std::vector<ArcInsertionMove> node_moves = possibleDeltaPlusMoves(ann_network,
                network.nodes_by_index[i], min_radius, max_radius);
        res.insert(std::end(res), std::begin(node_moves), std::end(node_moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanity(ann_network, res));
    return res;
}

void performMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move) {
    assert(checkSanity(ann_network, move));
    assert(move.moveType == MoveType::ArcInsertionMove || move.moveType == MoveType::DeltaPlusMove);
    assert(assertConsecutiveIndices(ann_network));
    assert(assertBranchLengths(ann_network));
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

    removeEdge(ann_network, network.edges_by_index[a_b_edge_index]);
    if (c_d_edge_index != a_b_edge_index) {
        removeEdge(ann_network, network.edges_by_index[c_d_edge_index]);
    }

    Edge *u_b_edge = addEdge(ann_network, u_b_link, to_b_link, u_b_edge_length[0],
            move.wanted_ub_pmatrix_index);
    Edge *v_d_edge = addEdge(ann_network, v_d_link, to_d_link, v_d_edge_length[0],
            move.wanted_vd_pmatrix_index);
    Edge *a_u_edge = addEdge(ann_network, from_a_link, to_u_link, a_u_edge_length[0],
            move.wanted_au_pmatrix_index);
    Edge *c_v_edge = addEdge(ann_network, from_c_link, v_c_link, c_v_edge_length[0],
            move.wanted_cv_pmatrix_index);
    Edge *u_v_edge = addEdge(ann_network, u_v_link, v_u_link, u_v_edge_length[0],
            move.wanted_uv_pmatrix_index);

    move.wanted_au_pmatrix_index = a_u_edge->pmatrix_index;
    move.wanted_cv_pmatrix_index = c_v_edge->pmatrix_index;
    move.wanted_ub_pmatrix_index = u_b_edge->pmatrix_index;
    move.wanted_vd_pmatrix_index = v_d_edge->pmatrix_index;
    move.wanted_uv_pmatrix_index = u_v_edge->pmatrix_index;

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

    set_edge_lengths(ann_network, u_b_edge->pmatrix_index, u_b_edge_length);
    set_edge_lengths(ann_network, v_d_edge->pmatrix_index, v_d_edge_length);
    set_edge_lengths(ann_network, a_u_edge->pmatrix_index, a_u_edge_length);
    set_edge_lengths(ann_network, c_v_edge->pmatrix_index, c_v_edge_length);
    set_edge_lengths(ann_network, u_v_edge->pmatrix_index, u_v_edge_length);

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
    checkSanity(network);
    assert(assertReticulationProbs(ann_network));
    assert(assertConsecutiveIndices(ann_network));
    assert(assertBranchLengths(ann_network));
}

void undoMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move) {
    assert(move.moveType == MoveType::ArcInsertionMove || move.moveType == MoveType::DeltaPlusMove);
    assert(assertConsecutiveIndices(ann_network));
    assert(assertBranchLengths(ann_network));
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
            move.a_u_len, move.a_b_len, move.c_d_len, move.v_d_len, move.u_b_len, MoveType::ArcRemovalMove, move.edge_orig_idx, move.node_orig_idx);
    removal.wanted_ab_pmatrix_index = move.ab_pmatrix_index;
    removal.wanted_cd_pmatrix_index = move.cd_pmatrix_index;
    removal.au_pmatrix_index = getEdgeTo(network, a, u)->pmatrix_index;
    removal.cv_pmatrix_index = getEdgeTo(network, c, v)->pmatrix_index;
    removal.uv_pmatrix_index = getEdgeTo(network, u, v)->pmatrix_index;
    removal.ub_pmatrix_index = getEdgeTo(network, u, b)->pmatrix_index;
    removal.vd_pmatrix_index = getEdgeTo(network, v, d)->pmatrix_index;
    performMove(ann_network, removal);
    assert(assertConsecutiveIndices(ann_network));
    assert(assertBranchLengths(ann_network));
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


}