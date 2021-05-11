#include "RSPRMove.hpp"

#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"

namespace netrax {

bool checkSanity(AnnotatedNetwork& ann_network, RSPRMove& move) {
    bool good = true;
    good &= (move.moveType == MoveType::RSPRMove || move.moveType == MoveType::RSPR1Move || move.moveType == MoveType::HeadMove || move.moveType == MoveType::TailMove);

    good &= (ann_network.network.nodes_by_index[move.x_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.x_prime_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.y_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.y_prime_clv_index] != nullptr);
    good &= (ann_network.network.nodes_by_index[move.z_clv_index] != nullptr);

    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.x_clv_index], ann_network.network.nodes_by_index[move.z_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.z_clv_index], ann_network.network.nodes_by_index[move.y_clv_index]));
    good &= (hasNeighbor(ann_network.network.nodes_by_index[move.x_prime_clv_index], ann_network.network.nodes_by_index[move.y_prime_clv_index]));

    return good;
}

bool checkSanity(AnnotatedNetwork& ann_network, std::vector<RSPRMove>& moves) {
    bool sane = true;
    for (size_t i = 0; i < moves.size(); ++i) {
        sane &= checkSanity(ann_network, moves[i]);
    }
    return sane;
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
        size_t y_clv_index, size_t z_clv_index, MoveType moveType, size_t edge_orig_idx, size_t node_orig_idx) {
    RSPRMove move = RSPRMove(edge_orig_idx, node_orig_idx);
    move.x_prime_clv_index = x_prime_clv_index;
    move.y_prime_clv_index = y_prime_clv_index;
    move.x_clv_index = x_clv_index;
    move.y_clv_index = y_clv_index;
    move.z_clv_index = z_clv_index;
    move.moveType = moveType;
    return move;
}

bool isRSPR1Move(RSPRMove& move) {
    return ((move.y_prime_clv_index == move.x_clv_index) 
        || (move.x_prime_clv_index == move.x_clv_index) 
        || (move.x_prime_clv_index == move.y_clv_index) 
        || (move.y_prime_clv_index == move.y_clv_index));
}

void possibleRSPRMovesInternal(std::vector<RSPRMove> &res, AnnotatedNetwork &ann_network, Node *x_prime,
        Node *y_prime, Node *x, Node *fixed_y, bool returnHead, bool returnTail,
        MoveType moveType, size_t edge_orig_idx, bool noRSPR1Moves) {
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

        size_t node_orig_idx = z->clv_index;

        if (z->type == NodeType::RETICULATION_NODE) { // head-moving rSPR move
            if (!hasPath(network, y_prime, w)) {
                RSPRMove move = buildRSPRMove(x_prime->clv_index, y_prime->clv_index, x->clv_index,
                        y->clv_index, z->clv_index, moveType, edge_orig_idx, node_orig_idx);
                move.x_z_len = get_edge_lengths(ann_network, getEdgeTo(network, x, z)->pmatrix_index);
                move.z_y_len = get_edge_lengths(ann_network, getEdgeTo(network, z, y)->pmatrix_index);
                move.x_prime_y_prime_len = get_edge_lengths(ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
                if (!noRSPR1Moves || !isRSPR1Move(move)) {
                    res.emplace_back(move);
                }
            }
        } else { // tail-moving rSPR move
            if (!hasPath(network, w, x_prime)) {
                RSPRMove move = buildRSPRMove(x_prime->clv_index, y_prime->clv_index, x->clv_index,
                        y->clv_index, z->clv_index, moveType, edge_orig_idx, node_orig_idx);
                move.x_z_len = get_edge_lengths(ann_network, getEdgeTo(network, x, z)->pmatrix_index);
                move.z_y_len = get_edge_lengths(ann_network, getEdgeTo(network, z, y)->pmatrix_index);
                move.x_prime_y_prime_len = get_edge_lengths(ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
                if (!noRSPR1Moves || !isRSPR1Move(move)) {
                    res.emplace_back(move);
                }
            }
        }
    }
}

void possibleRSPRMovesInternalNode(std::vector<RSPRMove> &res, AnnotatedNetwork &ann_network, Node *x,
        Node *y, Node* z, Node *x_prime, Node *fixed_y_prime, bool returnHead, bool returnTail,
        MoveType moveType, bool noRSPR1Moves, size_t min_radius, size_t max_radius) {
    Network &network = ann_network.network;

    std::vector<Node*> y_prime_cand;
    if (fixed_y_prime) {
        y_prime_cand.emplace_back(fixed_y_prime);
    } else {
        y_prime_cand = getChildren(network, x_prime);
    }

    for (Node* y_prime : y_prime_cand) {
        if (hasChild(network, x_prime, z) || hasChild(network, z, y_prime) || hasChild(network, x, y)) {
            continue;
        }
        bool problemFound = false;
        // if there is an arc z->w, we need that there is no w->x_prime path in the network
        std::vector<Node*> zChildren = getChildren(network, z);
        for (Node* w : zChildren) {
            if (problemFound) {
                break;
            }
            if (hasPath(network, w, x_prime)) {
                problemFound = true;
            }
        }
        // if there is an arc w->z, we need that there is no y_prime->w path in the network
        std::vector<Node*> zParents = getAllParents(network, z);
        for (Node* w : zParents) {
            if (problemFound) {
                break;
            }
            if (hasPath(network, y_prime, w)) {
                problemFound = true;
            }
        }

        if (problemFound) {
            continue;
        }

        size_t node_orig_idx = z->clv_index;

        size_t edge_orig_idx = getEdgeTo(network, x_prime, y_prime)->pmatrix_index;
        RSPRMove move = buildRSPRMove(x_prime->clv_index, y_prime->clv_index, x->clv_index,
                y->clv_index, z->clv_index, moveType, edge_orig_idx, node_orig_idx);
        move.x_z_len = get_edge_lengths(ann_network, getEdgeTo(network, x, z)->pmatrix_index);
        move.z_y_len = get_edge_lengths(ann_network, getEdgeTo(network, z, y)->pmatrix_index);
        move.x_prime_y_prime_len = get_edge_lengths(ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
        if (!noRSPR1Moves || !isRSPR1Move(move)) {
            res.emplace_back(move);
        }
    }
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, Node *node,
        Node *fixed_x_prime, Node *fixed_y_prime, MoveType moveType, bool noRSPR1Moves, bool returnHead = true, bool returnTail =
                true, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max()) {
    assert(node);
    if (node == ann_network.network.root || node->isTip()) {
        return {}; // because we need a parent and a child
    }

    Network &network = ann_network.network;
    std::vector<RSPRMove> res;
    std::vector<Node*> x_candidates = getAllParents(network, node);
    std::vector<Node*> y_candidates = getChildren(network, node);
    Node* z = node;

    for (Node* x : x_candidates) {
        for (Node* y : y_candidates) {
            if (fixed_x_prime) {
                possibleRSPRMovesInternalNode(res, ann_network, x, y, z, fixed_x_prime, fixed_y_prime, returnHead,
                        returnTail, moveType, noRSPR1Moves, min_radius, max_radius);
            } else {
                std::vector<Node*> radiusNodes = getNeighborsWithinRadius(network, node, min_radius, max_radius);
                for (Node* x_prime : radiusNodes) {
                    possibleRSPRMovesInternalNode(res, ann_network, x, y, z, x_prime, fixed_y_prime, returnHead,
                            returnTail, moveType, noRSPR1Moves, min_radius, max_radius);
                }
            }
        }
    }
    return res;
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge,
        Node *fixed_x, Node *fixed_y, MoveType moveType, bool noRSPR1Moves, bool returnHead = true, bool returnTail =
                true) {
    size_t edge_orig_idx = edge->pmatrix_index;
    Network &network = ann_network.network;
    std::vector<RSPRMove> res;
    Node *x_prime = getSource(network, edge);
    Node *y_prime = getTarget(network, edge);

    if (fixed_x) {
        possibleRSPRMovesInternal(res, ann_network, x_prime, y_prime, fixed_x, fixed_y, returnHead,
                returnTail, moveType, edge_orig_idx, noRSPR1Moves);
    } else {
        for (size_t i = 0; i < network.num_nodes(); ++i) {
            Node *x = network.nodes_by_index[i];
            possibleRSPRMovesInternal(res, ann_network, x_prime, y_prime, x, fixed_y, returnHead,
                    returnTail, moveType, edge_orig_idx, noRSPR1Moves);
        }
    }
    return res;
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves) {
    return possibleRSPRMoves(ann_network, edge, nullptr, nullptr, MoveType::RSPRMove, noRSPR1Moves, true, true);
}
std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves) {
    return possibleRSPRMoves(ann_network, edge, nullptr, nullptr, MoveType::TailMove, noRSPR1Moves, false, true);
}
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves) {
    return possibleRSPRMoves(ann_network, edge, nullptr, nullptr, MoveType::HeadMove, noRSPR1Moves, true, false);
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, Node *node, bool noRSPR1Moves, size_t min_radius, size_t max_radius) {
    return possibleRSPRMoves(ann_network, node, nullptr, nullptr, MoveType::RSPRMove, false, true, true, min_radius, max_radius);
}
std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, Node *node, bool noRSPR1Moves, size_t min_radius, size_t max_radius) {
    assert(node);
    if (node->type == NodeType::RETICULATION_NODE) { // we can only find head moves for z == node
        return {};
    }
    return possibleRSPRMoves(ann_network, node, nullptr, nullptr, MoveType::TailMove, false, true, true, min_radius, max_radius);
}
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, Node *node, bool noRSPR1Moves, size_t min_radius, size_t max_radius) {
    assert(node);
    if (node->type != NodeType::RETICULATION_NODE) { // we can only find tail moves for z == node
        return {};
    }
    return possibleRSPRMoves(ann_network, node, nullptr, nullptr, MoveType::HeadMove, false, true, true, min_radius, max_radius);
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const std::vector<Node*>& start_nodes, bool noRSPR1Moves, size_t min_radius, size_t max_radius) {
    std::vector<RSPRMove> res;
    for (Node* node : start_nodes) {
        std::vector<RSPRMove> res_node = possibleRSPRMoves(ann_network, node, noRSPR1Moves, min_radius, max_radius);
        res.insert(std::end(res), std::begin(res_node), std::end(res_node));
    }
    return res;
}

std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, const std::vector<Node*>& start_nodes, size_t min_radius, size_t max_radius) {
    std::vector<RSPRMove> res;
    for (Node* node : start_nodes) {
        std::vector<RSPRMove> res_node = possibleRSPR1Moves(ann_network, node, min_radius, max_radius);
        res.insert(std::end(res), std::begin(res_node), std::end(res_node));
    }
    return res;
}

std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, const std::vector<Node*>& start_nodes, bool noRSPR1Moves, size_t min_radius, size_t max_radius) {
    std::vector<RSPRMove> res;
    for (Node* node : start_nodes) {
        std::vector<RSPRMove> res_node = possibleTailMoves(ann_network, node, noRSPR1Moves, min_radius, max_radius);
        res.insert(std::end(res), std::begin(res_node), std::end(res_node));
    }
    return res;
}

std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, const std::vector<Node*>& start_nodes, bool noRSPR1Moves, size_t min_radius, size_t max_radius) {
    std::vector<RSPRMove> res;
    for (Node* node : start_nodes) {
        std::vector<RSPRMove> res_node = possibleHeadMoves(ann_network, node, noRSPR1Moves, min_radius, max_radius);
        res.insert(std::end(res), std::begin(res_node), std::end(res_node));
    }
    return res;
}


std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleTailMoves(ann_network, network.edges_by_index[i], noRSPR1Moves);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanity(ann_network, res));
    return res;
}

std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleHeadMoves(ann_network, network.edges_by_index[i], noRSPR1Moves);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanity(ann_network, res));
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
            MoveType::RSPR1Move, false);
    res.insert(std::end(res), std::begin(case1), std::end(case1));

// Case 2: x_prime == x
    std::vector<RSPRMove> case2 = possibleRSPRMoves(ann_network, edge, x_prime, nullptr,
            MoveType::RSPR1Move, false);
    res.insert(std::end(res), std::begin(case2), std::end(case2));

// Case 3: x_prime == y
    std::vector<RSPRMove> case3 = possibleRSPRMoves(ann_network, edge, nullptr, x_prime,
            MoveType::RSPR1Move, false);
    res.insert(std::end(res), std::begin(case3), std::end(case3));

// Case 4: y_prime == y
    std::vector<RSPRMove> case4 = possibleRSPRMoves(ann_network, edge, nullptr, y_prime,
            MoveType::RSPR1Move, false);
    res.insert(std::end(res), std::begin(case4), std::end(case4));

    return res;
}

std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, Node *node, size_t min_radius, size_t max_radius) {
    assert(node);
    if (node == ann_network.network.root || node->isTip()) {
        return {}; // because we need a parent and a child
    }
    Network &network = ann_network.network;
// in an rSPR1 move, either y_prime == x, x_prime == y, x_prime == x, or y_prime == y
    std::vector<RSPRMove> res;
    std::vector<Node*> x_candidates = getAllParents(network, node);
    std::vector<Node*> y_candidates = getChildren(network, node);

    std::vector<Node*> radiusNodes = getNeighborsWithinRadius(network, node, min_radius, max_radius); // these are the allowed nodes for x_prime

    for (Node* x : x_candidates) {
        for (Node * y : y_candidates) {
            // Case 1: y_prime == x
            std::vector<RSPRMove> case1 = possibleRSPRMoves(ann_network, node, nullptr, x,
                    MoveType::RSPR1Move, false, true, true, min_radius, max_radius);
            res.insert(std::end(res), std::begin(case1), std::end(case1));

            // Case 2: x_prime == x
            if (std::find(radiusNodes.begin(), radiusNodes.end(), x) != radiusNodes.end()) {
                std::vector<RSPRMove> case2 = possibleRSPRMoves(ann_network, node, x, nullptr,
                        MoveType::RSPR1Move, false, true, true, min_radius, max_radius);
                res.insert(std::end(res), std::begin(case2), std::end(case2));
            }

            // Case 3: x_prime == y
            if (std::find(radiusNodes.begin(), radiusNodes.end(), y) != radiusNodes.end()) {
                std::vector<RSPRMove> case3 = possibleRSPRMoves(ann_network, node, y, nullptr,
                        MoveType::RSPR1Move, false, true, true, min_radius, max_radius);
                res.insert(std::end(res), std::begin(case3), std::end(case3));
            }

            // Case 4: y_prime == y
            std::vector<RSPRMove> case4 = possibleRSPRMoves(ann_network, node, nullptr, y,
                    MoveType::RSPR1Move, false, true, true, min_radius, max_radius);
            res.insert(std::end(res), std::begin(case4), std::end(case4));
        }
    }

    return res;
}

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleRSPRMoves(ann_network, network.edges_by_index[i], noRSPR1Moves);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanity(ann_network, res));
    return res;
}

std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network) {
    std::vector<RSPRMove> res;
    Network &network = ann_network.network;
    for (size_t i = 0; i < network.num_branches(); ++i) {
        std::vector<RSPRMove> branch_moves = possibleRSPR1Moves(ann_network, network.edges_by_index[i]);
        res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
    }
    sortByProximity(res, ann_network);
    assert(checkSanity(ann_network, res));
    return res;
}

void performMove(AnnotatedNetwork &ann_network, RSPRMove &move) {
    assert(checkSanity(ann_network, move));
    assert(move.moveType == MoveType::RSPRMove || move.moveType == MoveType::RSPR1Move || move.moveType == MoveType::HeadMove || move.moveType == MoveType::TailMove);
    assert(assertConsecutiveIndices(ann_network));
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

    x_out_link->edge_pmatrix_index = x_y_edge->pmatrix_index;
    y_in_link->edge_pmatrix_index = x_y_edge->pmatrix_index;
    x_prime_out_link->edge_pmatrix_index = x_prime_z_edge->pmatrix_index;
    z_in_link->edge_pmatrix_index = x_prime_z_edge->pmatrix_index;
    z_out_link->edge_pmatrix_index = z_y_prime_edge->pmatrix_index;
    y_prime_in_link->edge_pmatrix_index = z_y_prime_edge->pmatrix_index;

    set_edge_lengths(ann_network, x_y_edge->pmatrix_index, x_y_len);
    set_edge_lengths(ann_network, x_prime_z_edge->pmatrix_index, x_prime_z_len);
    set_edge_lengths(ann_network, z_y_prime_edge->pmatrix_index, z_y_prime_len);

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

    //std::cout << exportDebugInfo(ann_network.network) << "\n";

    assert(assertReticulationProbs(ann_network));
    assert(assertConsecutiveIndices(ann_network));
}

void undoMove(AnnotatedNetwork &ann_network, RSPRMove &move) {
    assert(move.moveType == MoveType::RSPRMove || move.moveType == MoveType::RSPR1Move || move.moveType == MoveType::HeadMove || move.moveType == MoveType::TailMove);
    assert(assertConsecutiveIndices(ann_network));
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

    x_prime_out_link->edge_pmatrix_index = x_prime_y_prime_edge->pmatrix_index;
    y_prime_in_link->edge_pmatrix_index = x_prime_y_prime_edge->pmatrix_index;
    x_out_link->edge_pmatrix_index = x_z_edge->pmatrix_index;
    z_in_link->edge_pmatrix_index = x_z_edge->pmatrix_index;
    z_out_link->edge_pmatrix_index = z_y_edge->pmatrix_index;
    y_in_link->edge_pmatrix_index = z_y_edge->pmatrix_index;

    set_edge_lengths(ann_network, x_prime_y_prime_edge->pmatrix_index, x_prime_y_prime_len);
    set_edge_lengths(ann_network, x_z_edge->pmatrix_index, x_z_len);
    set_edge_lengths(ann_network, z_y_edge->pmatrix_index, z_y_len);

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
    assert(assertReticulationProbs(ann_network));
    assert(assertConsecutiveIndices(ann_network));
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