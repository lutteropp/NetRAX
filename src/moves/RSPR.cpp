#include "RSPR.hpp"
#include "Move.hpp"

#include "../DebugPrintFunctions.hpp"
#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"
#include "GeneralMoveFunctions.hpp"

namespace netrax {

bool checkSanityRSPR(AnnotatedNetwork &ann_network, const Move &move) {
  bool good = true;
  good &= (move.moveType == MoveType::RSPRMove ||
           move.moveType == MoveType::RSPR1Move ||
           move.moveType == MoveType::HeadMove ||
           move.moveType == MoveType::TailMove);

  Node *x = ann_network.network.nodes_by_index[move.rsprData.x_clv_index];
  Node *y = ann_network.network.nodes_by_index[move.rsprData.y_clv_index];
  Node *z = ann_network.network.nodes_by_index[move.rsprData.z_clv_index];
  Node *x_prime =
      ann_network.network.nodes_by_index[move.rsprData.x_prime_clv_index];
  Node *y_prime =
      ann_network.network.nodes_by_index[move.rsprData.y_prime_clv_index];

  if (good) good &= (x != nullptr);
  if (good) good &= (y != nullptr);
  if (good) good &= (z != nullptr);
  if (good) good &= (x_prime != nullptr);
  if (good) good &= (y_prime != nullptr);

  if (good) good &= (hasChild(ann_network.network, x, z));
  if (good) good &= (hasChild(ann_network.network, z, y));
  if (good) good &= (hasChild(ann_network.network, x_prime, y_prime));

  if (good) good &= (!hasChild(ann_network.network, x_prime, z));
  if (good) good &= (!hasChild(ann_network.network, z, x_prime));
  if (good) good &= (!hasChild(ann_network.network, x, y));

  if (move.moveType == MoveType::HeadMove) {
    if (good) good &= (z->getType() == NodeType::RETICULATION_NODE);
  }
  if (move.moveType == MoveType::TailMove) {
    if (good) good &= (z->getType() != NodeType::RETICULATION_NODE);
  }
  if (move.moveType == MoveType::RSPR1Move) {
    if (good) good &= ((x == y_prime) || (y == x_prime) || (y == y_prime));
  }

  // the resulting network needs to still be acyclic
  if (good) {
    if (z->getType() == NodeType::RETICULATION_NODE) { // head-moving
      Node* w = getReticulationOtherParent(ann_network.network, z, x);
      assert(w);
      good &= (!hasPath(ann_network.network, w, x_prime));
    } else { // tail-moving
      Node* w = getOtherChild(ann_network.network, z, y);
      assert(w);
      good &= (!hasPath(ann_network.network, y_prime, w));
    }
  }

  return good;
}

bool checkSanityRSPR(AnnotatedNetwork &ann_network, std::vector<Move> &moves) {
  bool sane = true;
  for (size_t i = 0; i < moves.size(); ++i) {
    sane &= checkSanityRSPR(ann_network, moves[i]);
  }
  return sane;
}

std::vector<std::pair<const Node *, const Node *>> getZYChoices(
    Network &network, const Node *x_prime, const Node *y_prime, const Node *x,
    const Node *fixed_y = nullptr, bool returnHead = true,
    bool returnTail = true) {
  std::vector<std::pair<const Node *, const Node *>> res;
  auto x_prime_children = getChildren(network, x_prime);
  auto x_children = getChildren(network, x);
  for (const Node *z : x_children) {
    if (std::find(x_prime_children.begin(), x_prime_children.end(), z) !=
        x_prime_children.end()) {
      continue;
    }
    if (!returnHead &&
        z->type == NodeType::RETICULATION_NODE) {  // head-moving rSPR move
      continue;
    }
    if (!returnTail &&
        z->type != NodeType::RETICULATION_NODE) {  // tail-moving rSPR move
      continue;
    }
    auto z_children = getChildren(network, z);
    if (std::find(z_children.begin(), z_children.end(), y_prime) !=
        z_children.end()) {
      continue;
    }

    for (const Node *y : z_children) {
      if (fixed_y && y != fixed_y) {
        continue;
      }
      if (std::find(x_children.begin(), x_children.end(), y) !=
          x_children.end()) {
        continue;
      }
      assert(hasNeighbor(x, z));
      assert(hasNeighbor(z, y));
      res.emplace_back(std::make_pair(z, y));
    }
  }
  return res;
}

Move buildMoveRSPR(Network &network, size_t x_prime_clv_index,
                   size_t y_prime_clv_index, size_t x_clv_index,
                   size_t y_clv_index, size_t z_clv_index, MoveType moveType,
                   size_t edge_orig_idx, size_t node_orig_idx) {
  Move move = Move(moveType, edge_orig_idx, node_orig_idx);
  move.rsprData.x_prime_clv_index = x_prime_clv_index;
  move.rsprData.y_prime_clv_index = y_prime_clv_index;
  move.rsprData.x_clv_index = x_clv_index;
  move.rsprData.y_clv_index = y_clv_index;
  move.rsprData.z_clv_index = z_clv_index;
  move.moveType = moveType;

  if (network.nodes_by_index[y_prime_clv_index]->getType() ==
      NodeType::RETICULATION_NODE) {
    move.rsprData.y_prime_first_parent_clv_index =
        getReticulationFirstParent(network,
                                   network.nodes_by_index[y_prime_clv_index])
            ->clv_index;
  }
  if (network.nodes_by_index[y_clv_index]->getType() ==
      NodeType::RETICULATION_NODE) {
    move.rsprData.y_first_parent_clv_index =
        getReticulationFirstParent(network, network.nodes_by_index[y_clv_index])
            ->clv_index;
  }
  if (network.nodes_by_index[z_clv_index]->getType() ==
      NodeType::RETICULATION_NODE) {
    move.rsprData.z_first_parent_clv_index =
        getReticulationFirstParent(network, network.nodes_by_index[z_clv_index])
            ->clv_index;
  }
  return move;
}

bool isRSPR1Move(Move &move) {
  return ((move.rsprData.y_prime_clv_index == move.rsprData.x_clv_index) ||
          (move.rsprData.x_prime_clv_index == move.rsprData.x_clv_index) ||
          (move.rsprData.x_prime_clv_index == move.rsprData.y_clv_index) ||
          (move.rsprData.y_prime_clv_index == move.rsprData.y_clv_index));
}

void possibleMovesRSPRInternal(
    std::vector<Move> &res, AnnotatedNetwork &ann_network, const Node *x_prime,
    const Node *y_prime, const Node *x, const Node *fixed_y, bool returnHead,
    bool returnTail, MoveType moveType, size_t edge_orig_idx, bool noRSPR1Moves,
    int min_radius, int max_radius) {
  Network &network = ann_network.network;

  auto zy = getZYChoices(network, x_prime, y_prime, x, fixed_y, returnHead,
                         returnTail);
  for (const auto &entry : zy) {
    const Node *z = entry.first;
    const Node *y = entry.second;

    std::vector<Node *> radiusNodes = getNeighborsWithinRadius(
        network, z, min_radius,
        max_radius);  // these are the allowed nodes for x_prime
    if (std::find(radiusNodes.begin(), radiusNodes.end(), x_prime) ==
        radiusNodes.end()) {
      continue;
    }

    const Node *w = nullptr;
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

    if (z->type == NodeType::RETICULATION_NODE) {  // head-moving rSPR move
      if (!hasPath(network, y_prime, w)) {
        Move move = buildMoveRSPR(
            network, x_prime->clv_index, y_prime->clv_index, x->clv_index,
            y->clv_index, z->clv_index, moveType, edge_orig_idx, node_orig_idx);
        move.rsprData.x_z_len = get_edge_lengths(
            ann_network, getEdgeTo(network, x, z)->pmatrix_index);
        move.rsprData.z_y_len = get_edge_lengths(
            ann_network, getEdgeTo(network, z, y)->pmatrix_index);
        move.rsprData.x_prime_y_prime_len = get_edge_lengths(
            ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
        if (!noRSPR1Moves || !isRSPR1Move(move)) {
          res.emplace_back(move);
        }
      }
    } else {  // tail-moving rSPR move
      if (!hasPath(network, w, x_prime)) {
        Move move = buildMoveRSPR(
            network, x_prime->clv_index, y_prime->clv_index, x->clv_index,
            y->clv_index, z->clv_index, moveType, edge_orig_idx, node_orig_idx);
        move.rsprData.x_z_len = get_edge_lengths(
            ann_network, getEdgeTo(network, x, z)->pmatrix_index);
        move.rsprData.z_y_len = get_edge_lengths(
            ann_network, getEdgeTo(network, z, y)->pmatrix_index);
        move.rsprData.x_prime_y_prime_len = get_edge_lengths(
            ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
        if (!noRSPR1Moves || !isRSPR1Move(move)) {
          res.emplace_back(move);
        }
      }
    }
  }
}

void possibleMovesRSPRInternalNode(
    std::vector<Move> &res, AnnotatedNetwork &ann_network, const Node *x,
    const Node *y, const Node *z, const Node *x_prime,
    const Node *fixed_y_prime, bool returnHead, bool returnTail,
    MoveType moveType, bool noRSPR1Moves, int min_radius, int max_radius) {
  Network &network = ann_network.network;

  std::vector<Node *> y_prime_cand;
  if (fixed_y_prime) {
    y_prime_cand.emplace_back(
        ann_network.network.nodes_by_index[fixed_y_prime->clv_index]);
  } else {
    y_prime_cand = getChildren(network, x_prime);
  }

  for (const Node *y_prime : y_prime_cand) {
    if (hasChild(network, x_prime, z) || hasChild(network, z, y_prime) ||
        hasChild(network, x, y)) {
      continue;
    }
    bool problemFound = false;

    if (z->getType() == NodeType::RETICULATION_NODE) { // head-moving
      Node* w = getReticulationOtherParent(ann_network.network, z, x);
      assert(w);
      problemFound |= (!hasPath(ann_network.network, w, x_prime));
    } else { // tail-moving
      Node* w = getOtherChild(ann_network.network, z, y);
      assert(w);
      problemFound |= (!hasPath(ann_network.network, y_prime, w));
    }

    if (problemFound) {
      continue;
    }

    if ((z->type == NodeType::RETICULATION_NODE && returnHead) ||
        (z->type != NodeType::RETICULATION_NODE && returnTail)) {
      size_t node_orig_idx = z->clv_index;

      size_t edge_orig_idx =
          getEdgeTo(network, x_prime, y_prime)->pmatrix_index;
      Move move = buildMoveRSPR(network, x_prime->clv_index, y_prime->clv_index,
                                x->clv_index, y->clv_index, z->clv_index,
                                moveType, edge_orig_idx, node_orig_idx);
      move.rsprData.x_z_len = get_edge_lengths(
          ann_network, getEdgeTo(network, x, z)->pmatrix_index);
      move.rsprData.z_y_len = get_edge_lengths(
          ann_network, getEdgeTo(network, z, y)->pmatrix_index);
      move.rsprData.x_prime_y_prime_len = get_edge_lengths(
          ann_network, getEdgeTo(network, x_prime, y_prime)->pmatrix_index);
      if (!noRSPR1Moves || !isRSPR1Move(move)) {
        res.emplace_back(move);
      }
    }
  }
}

std::vector<Move> possibleMovesRSPR(
    AnnotatedNetwork &ann_network, const Node *node, const Node *fixed_x_prime,
    const Node *fixed_y_prime, MoveType moveType, bool noRSPR1Moves,
    bool returnHead = true, bool returnTail = true, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max()) {
  assert(node);
  if (node == ann_network.network.root || node->isTip()) {
    return {};  // because we need a parent and a child
  }

  Network &network = ann_network.network;
  std::vector<Move> res;
  std::vector<Node *> x_candidates = getAllParents(network, node);
  std::vector<Node *> y_candidates = getChildren(network, node);
  const Node *z = node;

  for (const Node *x : x_candidates) {
    for (const Node *y : y_candidates) {
      if (fixed_x_prime) {
        possibleMovesRSPRInternalNode(
            res, ann_network, x, y, z, fixed_x_prime, fixed_y_prime, returnHead,
            returnTail, moveType, noRSPR1Moves, min_radius, max_radius);
      } else {
        std::vector<Node *> radiusNodes =
            getNeighborsWithinRadius(network, node, min_radius, max_radius);
        for (const Node *x_prime : radiusNodes) {
          possibleMovesRSPRInternalNode(
              res, ann_network, x, y, z, x_prime, fixed_y_prime, returnHead,
              returnTail, moveType, noRSPR1Moves, min_radius, max_radius);
        }
      }
    }
  }
  return res;
}

std::vector<Move> possibleMovesRSPR(
    AnnotatedNetwork &ann_network, const Edge *edge, const Node *fixed_x,
    const Node *fixed_y, MoveType moveType, bool noRSPR1Moves,
    bool returnHead = true, bool returnTail = true, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max()) {
  size_t edge_orig_idx = edge->pmatrix_index;
  Network &network = ann_network.network;
  std::vector<Move> res;
  const Node *x_prime = getSource(network, edge);
  const Node *y_prime = getTarget(network, edge);

  if (fixed_x) {
    possibleMovesRSPRInternal(res, ann_network, x_prime, y_prime, fixed_x,
                              fixed_y, returnHead, returnTail, moveType,
                              edge_orig_idx, noRSPR1Moves, min_radius,
                              max_radius);
  } else {
    for (size_t i = 0; i < network.num_nodes(); ++i) {
      Node *x = network.nodes_by_index[i];
      possibleMovesRSPRInternal(res, ann_network, x_prime, y_prime, x, fixed_y,
                                returnHead, returnTail, moveType, edge_orig_idx,
                                noRSPR1Moves, min_radius, max_radius);
    }
  }
  return res;
}

std::vector<Move> possibleMovesRSPR(AnnotatedNetwork &ann_network,
                                    const Edge *edge, bool noRSPR1Moves,
                                    int min_radius, int max_radius) {
  return possibleMovesRSPR(ann_network, edge, nullptr, nullptr,
                           MoveType::RSPRMove, noRSPR1Moves, true, true,
                           min_radius, max_radius);
}
std::vector<Move> possibleMovesTail(AnnotatedNetwork &ann_network,
                                    const Edge *edge, bool noRSPR1Moves,
                                    int min_radius, int max_radius) {
  return possibleMovesRSPR(ann_network, edge, nullptr, nullptr,
                           MoveType::TailMove, noRSPR1Moves, false, true,
                           min_radius, max_radius);
}
std::vector<Move> possibleMovesHead(AnnotatedNetwork &ann_network,
                                    const Edge *edge, bool noRSPR1Moves,
                                    int min_radius, int max_radius) {
  return possibleMovesRSPR(ann_network, edge, nullptr, nullptr,
                           MoveType::HeadMove, noRSPR1Moves, true, false,
                           min_radius, max_radius);
}

std::vector<Move> possibleMovesRSPR(AnnotatedNetwork &ann_network,
                                    const Node *node, bool noRSPR1Moves,
                                    int min_radius, int max_radius) {
  return possibleMovesRSPR(ann_network, node, nullptr, nullptr,
                           MoveType::RSPRMove, noRSPR1Moves, true, true,
                           min_radius, max_radius);
}
std::vector<Move> possibleMovesTail(AnnotatedNetwork &ann_network,
                                    const Node *node, bool noRSPR1Moves,
                                    int min_radius, int max_radius) {
  assert(node);
  if (node->type == NodeType::RETICULATION_NODE) {  // we can only find head
                                                    // moves for z == node
    return {};
  }
  return possibleMovesRSPR(ann_network, node, nullptr, nullptr,
                           MoveType::TailMove, noRSPR1Moves, true, true,
                           min_radius, max_radius);
}
std::vector<Move> possibleMovesHead(AnnotatedNetwork &ann_network,
                                    const Node *node, bool noRSPR1Moves,
                                    int min_radius, int max_radius) {
  assert(node);
  if (node->type != NodeType::RETICULATION_NODE) {  // we can only find tail
                                                    // moves for z == node
    return {};
  }
  return possibleMovesRSPR(ann_network, node, nullptr, nullptr,
                           MoveType::HeadMove, noRSPR1Moves, true, true,
                           min_radius, max_radius);
}

std::vector<Move> possibleMovesRSPR(AnnotatedNetwork &ann_network,
                                    const std::vector<Node *> &start_nodes,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  for (const Node *node : start_nodes) {
    std::vector<Move> res_node = possibleMovesRSPR(
        ann_network, node, noRSPR1Moves, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_node), std::end(res_node));
  }
  return res;
}

std::vector<Move> possibleMovesRSPR1(AnnotatedNetwork &ann_network,
                                     const std::vector<Node *> &start_nodes,
                                     int min_radius, int max_radius) {
  std::vector<Move> res;
  for (Node *node : start_nodes) {
    std::vector<Move> res_node =
        possibleMovesRSPR1(ann_network, node, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_node), std::end(res_node));
  }
  return res;
}

std::vector<Move> possibleMovesTail(AnnotatedNetwork &ann_network,
                                    const std::vector<Node *> &start_nodes,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  for (Node *node : start_nodes) {
    std::vector<Move> res_node = possibleMovesTail(
        ann_network, node, noRSPR1Moves, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_node), std::end(res_node));
  }
  return res;
}

std::vector<Move> possibleMovesHead(AnnotatedNetwork &ann_network,
                                    const std::vector<Node *> &start_nodes,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  for (Node *node : start_nodes) {
    std::vector<Move> res_node = possibleMovesHead(
        ann_network, node, noRSPR1Moves, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_node), std::end(res_node));
  }
  return res;
}

std::vector<Move> possibleMovesRSPR(AnnotatedNetwork &ann_network,
                                    const std::vector<Edge *> &start_edges,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  for (const Edge *edge : start_edges) {
    std::vector<Move> res_edge = possibleMovesRSPR(
        ann_network, edge, noRSPR1Moves, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_edge), std::end(res_edge));
  }
  return res;
}

std::vector<Move> possibleMovesRSPR1(AnnotatedNetwork &ann_network,
                                     const std::vector<Edge *> &start_edges,
                                     int min_radius, int max_radius) {
  std::vector<Move> res;
  for (Edge *edge : start_edges) {
    std::vector<Move> res_edge =
        possibleMovesRSPR1(ann_network, edge, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_edge), std::end(res_edge));
  }
  return res;
}

std::vector<Move> possibleMovesTail(AnnotatedNetwork &ann_network,
                                    const std::vector<Edge *> &start_edges,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  for (Edge *edge : start_edges) {
    std::vector<Move> res_edge = possibleMovesTail(
        ann_network, edge, noRSPR1Moves, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_edge), std::end(res_edge));
  }
  return res;
}

std::vector<Move> possibleMovesHead(AnnotatedNetwork &ann_network,
                                    const std::vector<Edge *> &start_edges,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  for (Edge *edge : start_edges) {
    std::vector<Move> res_edge = possibleMovesHead(
        ann_network, edge, noRSPR1Moves, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_edge), std::end(res_edge));
  }
  return res;
}

std::vector<Move> possibleMovesTail(AnnotatedNetwork &ann_network,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  Network &network = ann_network.network;
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    std::vector<Move> branch_moves =
        possibleMovesTail(ann_network, network.nodes_by_index[i], noRSPR1Moves,
                          min_radius, max_radius);
    res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
  }
  sortByProximity(res, ann_network);
  assert(checkSanityRSPR(ann_network, res));
  return res;
}

std::vector<Move> possibleMovesHead(AnnotatedNetwork &ann_network,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  Network &network = ann_network.network;
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    std::vector<Move> branch_moves =
        possibleMovesHead(ann_network, network.nodes_by_index[i], noRSPR1Moves,
                          min_radius, max_radius);
    res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
  }
  sortByProximity(res, ann_network);
  assert(checkSanityRSPR(ann_network, res));
  return res;
}

std::vector<Move> possibleMovesRSPR1(AnnotatedNetwork &ann_network,
                                     const Edge *edge, int min_radius,
                                     int max_radius) {
  assert(edge);
  std::vector<Move> res;
  std::vector<Move> rsprMoves =
      possibleMovesRSPR(ann_network, edge, false, min_radius, max_radius);
  for (size_t i = 0; i < rsprMoves.size(); ++i) {
    if (isRSPR1Move(rsprMoves[i])) {
      res.emplace_back(rsprMoves[i]);
    }
  }
  return res;
}

std::vector<Move> possibleMovesRSPR1(AnnotatedNetwork &ann_network,
                                     const Node *node, int min_radius,
                                     int max_radius) {
  assert(node);
  if (node == ann_network.network.root || node->isTip()) {
    return {};  // because we need a parent and a child
  }
  Network &network = ann_network.network;
  // in an rSPR1 move, either y_prime == x, x_prime == y, x_prime == x, or
  // y_prime == y
  std::vector<Move> res;
  std::vector<Node *> x_candidates = getAllParents(network, node);
  std::vector<Node *> y_candidates = getChildren(network, node);

  std::vector<Node *> radiusNodes = getNeighborsWithinRadius(
      network, node, min_radius,
      max_radius);  // these are the allowed nodes for x_prime

  for (const Node *x : x_candidates) {
    for (const Node *y : y_candidates) {
      // Case 1: y_prime == x
      std::vector<Move> case1 =
          possibleMovesRSPR(ann_network, node, nullptr, x, MoveType::RSPR1Move,
                            false, true, true, min_radius, max_radius);
      res.insert(std::end(res), std::begin(case1), std::end(case1));

      // Case 2: x_prime == x
      if (std::find(radiusNodes.begin(), radiusNodes.end(), x) !=
          radiusNodes.end()) {
        std::vector<Move> case2 = possibleMovesRSPR(
            ann_network, node, x, nullptr, MoveType::RSPR1Move, false, true,
            true, min_radius, max_radius);
        res.insert(std::end(res), std::begin(case2), std::end(case2));
      }

      // Case 3: x_prime == y
      if (std::find(radiusNodes.begin(), radiusNodes.end(), y) !=
          radiusNodes.end()) {
        std::vector<Move> case3 = possibleMovesRSPR(
            ann_network, node, y, nullptr, MoveType::RSPR1Move, false, true,
            true, min_radius, max_radius);
        res.insert(std::end(res), std::begin(case3), std::end(case3));
      }

      // Case 4: y_prime == y
      std::vector<Move> case4 =
          possibleMovesRSPR(ann_network, node, nullptr, y, MoveType::RSPR1Move,
                            false, true, true, min_radius, max_radius);
      res.insert(std::end(res), std::begin(case4), std::end(case4));
    }
  }

  return res;
}

std::vector<Move> possibleMovesRSPR(AnnotatedNetwork &ann_network,
                                    bool noRSPR1Moves, int min_radius,
                                    int max_radius) {
  std::vector<Move> res;
  Network &network = ann_network.network;
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    std::vector<Move> branch_moves =
        possibleMovesRSPR(ann_network, network.nodes_by_index[i], noRSPR1Moves,
                          min_radius, max_radius);
    res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
  }
  sortByProximity(res, ann_network);
  assert(checkSanityRSPR(ann_network, res));
  return res;
}

std::vector<Move> possibleMovesRSPR1(AnnotatedNetwork &ann_network,
                                     int min_radius, int max_radius) {
  std::vector<Move> res;
  Network &network = ann_network.network;
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    std::vector<Move> branch_moves = possibleMovesRSPR1(
        ann_network, network.nodes_by_index[i], min_radius, max_radius);
    res.insert(std::end(res), std::begin(branch_moves), std::end(branch_moves));
  }
  sortByProximity(res, ann_network);
  assert(checkSanityRSPR(ann_network, res));
  return res;
}

std::vector<Move> possibleMovesRSPR(AnnotatedNetwork &ann_network,
                                    const std::vector<Node *> &start_nodes,
                                    int min_radius, int max_radius) {
  return possibleMovesRSPR(ann_network, start_nodes, min_radius, max_radius);
}

void performMoveRSPR(AnnotatedNetwork &ann_network, Move &move) {
  assert(checkSanityRSPR(ann_network, move));
  assert(move.moveType == MoveType::RSPRMove ||
         move.moveType == MoveType::RSPR1Move ||
         move.moveType == MoveType::HeadMove ||
         move.moveType == MoveType::TailMove);
  assert(assertConsecutiveIndices(ann_network));
  Network &network = ann_network.network;
  assert(checkSanity(ann_network));
  Node *x_prime = network.nodes_by_index[move.rsprData.x_prime_clv_index];
  Node *y_prime = network.nodes_by_index[move.rsprData.y_prime_clv_index];
  Node *x = network.nodes_by_index[move.rsprData.x_clv_index];
  Node *y = network.nodes_by_index[move.rsprData.y_clv_index];
  Node *z = network.nodes_by_index[move.rsprData.z_clv_index];

  Link *x_out_link = getLinkToNode(network, x, z);
  Link *z_in_link = getLinkToNode(network, z, x);
  Link *z_out_link = getLinkToNode(network, z, y);
  Link *x_prime_out_link = getLinkToNode(network, x_prime, y_prime);
  Link *y_prime_in_link = getLinkToNode(network, y_prime, x_prime);
  Link *y_in_link = getLinkToNode(network, y, z);

  Edge *x_z_edge = getEdgeTo(network, x, z);
  Edge *z_y_edge = getEdgeTo(network, z, y);
  Edge *x_prime_y_prime_edge = getEdgeTo(network, x_prime, y_prime);

  std::vector<double> x_z_len =
      get_edge_lengths(ann_network, x_z_edge->pmatrix_index);
  std::vector<double> z_y_len =
      get_edge_lengths(ann_network, z_y_edge->pmatrix_index);
  std::vector<double> x_prime_y_prime_len =
      get_edge_lengths(ann_network, x_prime_y_prime_edge->pmatrix_index);

  double min_br = ann_network.options.brlen_min;
  double max_br = ann_network.options.brlen_max;

  size_t n_p =
      (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED)
          ? ann_network.fake_treeinfo->partition_count
          : 1;
  std::vector<double> x_y_len(n_p), x_prime_z_len(n_p), z_y_prime_len(n_p);
  for (size_t p = 0; p < n_p; ++p) {
    x_y_len[p] = std::min(x_z_len[p] + z_y_len[p], max_br);
    assert(x_y_len[p] >= min_br);
    x_prime_z_len[p] = std::max(x_prime_y_prime_len[p] / 2, min_br);
    assert(x_prime_z_len[p] <= max_br);
    z_y_prime_len[p] = std::max(x_prime_y_prime_len[p] / 2, min_br);
    assert(x_prime_y_prime_len[p] <= max_br);
  }

  assert(x_prime_out_link->edge_pmatrix_index ==
         x_prime_y_prime_edge->pmatrix_index);
  assert(y_prime_in_link->edge_pmatrix_index ==
         x_prime_y_prime_edge->pmatrix_index);
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

  repairReticulationData(network);

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

  // std::cout << exportDebugInfo(ann_network.network) << "\n";

  assert(assertReticulationProbs(ann_network));
  assert(assertConsecutiveIndices(ann_network));
  assert(checkSanity(ann_network));
}

void undoMoveRSPR(AnnotatedNetwork &ann_network, Move &move) {
  assert(move.moveType == MoveType::RSPRMove ||
         move.moveType == MoveType::RSPR1Move ||
         move.moveType == MoveType::HeadMove ||
         move.moveType == MoveType::TailMove);
  assert(assertConsecutiveIndices(ann_network));
  Network &network = ann_network.network;
  assert(checkSanity(ann_network));
  Node *x_prime = network.nodes_by_index[move.rsprData.x_prime_clv_index];
  Node *y_prime = network.nodes_by_index[move.rsprData.y_prime_clv_index];
  Node *x = network.nodes_by_index[move.rsprData.x_clv_index];
  Node *y = network.nodes_by_index[move.rsprData.y_clv_index];
  Node *z = network.nodes_by_index[move.rsprData.z_clv_index];

  Link *x_out_link = getLinkToNode(network, x, y);
  Link *z_in_link = getLinkToNode(network, z, x_prime);
  Link *z_out_link = getLinkToNode(network, z, y_prime);
  Link *x_prime_out_link = getLinkToNode(network, x_prime, z);
  Link *y_prime_in_link = getLinkToNode(network, y_prime, z);
  Link *y_in_link = getLinkToNode(network, y, x);

  Edge *x_y_edge = getEdgeTo(network, x, y);
  Edge *x_prime_z_edge = getEdgeTo(network, x_prime, z);
  Edge *z_y_prime_edge = getEdgeTo(network, z, y_prime);

  std::vector<double> x_z_len = move.rsprData.x_z_len;
  std::vector<double> z_y_len = move.rsprData.z_y_len;
  std::vector<double> x_prime_y_prime_len = move.rsprData.x_prime_y_prime_len;

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

  set_edge_lengths(ann_network, x_prime_y_prime_edge->pmatrix_index,
                   x_prime_y_prime_len);
  set_edge_lengths(ann_network, x_z_edge->pmatrix_index, x_z_len);
  set_edge_lengths(ann_network, z_y_edge->pmatrix_index, z_y_len);

  repairReticulationData(network);

  if (y_prime->getType() == NodeType::RETICULATION_NODE) {
    if (getReticulationFirstParent(network, y_prime)->clv_index !=
        move.rsprData.y_prime_first_parent_clv_index) {
      assert(getReticulationSecondParent(network, y_prime)->clv_index ==
             move.rsprData.y_prime_first_parent_clv_index);
      std::swap(y_prime->getReticulationData()->link_to_first_parent,
                y_prime->getReticulationData()->link_to_second_parent);
    }
  }
  if (y->getType() == NodeType::RETICULATION_NODE) {
    if (getReticulationFirstParent(network, y)->clv_index !=
        move.rsprData.y_first_parent_clv_index) {
      assert(getReticulationSecondParent(network, y)->clv_index ==
             move.rsprData.y_first_parent_clv_index);
      std::swap(y->getReticulationData()->link_to_first_parent,
                y->getReticulationData()->link_to_second_parent);
    }
  }
  if (z->getType() == NodeType::RETICULATION_NODE) {
    if (getReticulationFirstParent(network, z)->clv_index !=
        move.rsprData.z_first_parent_clv_index) {
      assert(getReticulationSecondParent(network, z)->clv_index ==
             move.rsprData.z_first_parent_clv_index);
      std::swap(z->getReticulationData()->link_to_first_parent,
                z->getReticulationData()->link_to_second_parent);
    }
  }

  std::vector<bool> visited(network.nodes.size(), false);
  invalidateHigherCLVs(ann_network, z, false, visited);
  invalidateHigherCLVs(ann_network, x, false, visited);
  invalidateHigherCLVs(ann_network, x_prime, false, visited);
  invalidateHigherCLVs(ann_network, y, false, visited);
  invalidateHigherCLVs(ann_network, y_prime, false, visited);
  invalidatePmatrixIndex(ann_network, x_prime_y_prime_edge->pmatrix_index,
                         visited);
  invalidatePmatrixIndex(ann_network, x_z_edge->pmatrix_index, visited);
  invalidatePmatrixIndex(ann_network, z_y_edge->pmatrix_index, visited);

  ann_network.travbuffer = reversed_topological_sort(ann_network.network);
  assert(assertReticulationProbs(ann_network));
  assert(assertConsecutiveIndices(ann_network));
  assert(checkSanity(ann_network));
}

std::string toStringRSPR(const Move &move) {
  std::stringstream ss;
  ss << "rSPR move:\n";
  ss << "  x_prime = " << move.rsprData.x_prime_clv_index << "\n";
  ss << "  y_prime = " << move.rsprData.y_prime_clv_index << "\n";
  ss << "  x = " << move.rsprData.x_clv_index << "\n";
  ss << "  y = " << move.rsprData.y_clv_index << "\n";
  ss << "  z = " << move.rsprData.z_clv_index << "\n";
  ss << "  y_prime_first_parent_clv_index = "
     << move.rsprData.y_prime_first_parent_clv_index << "\n";
  ss << "  y_first_parent_clv_index = "
     << move.rsprData.y_first_parent_clv_index << "\n";
  ss << "  z_first_parent_clv_index = "
     << move.rsprData.z_first_parent_clv_index << "\n";
  ss << "  remapped_clv_indices: " << move.remapped_clv_indices << "\n";
  ss << "  remapped_pmatrix_indices: " << move.remapped_pmatrix_indices << "\n";
  ss << "  remapped_reticulation_indices: "
     << move.remapped_reticulation_indices << "\n";
  return ss.str();
}

std::unordered_set<size_t> brlenOptCandidatesRSPR(AnnotatedNetwork &ann_network,
                                                  const Move &move) {
  Node *x = ann_network.network.nodes_by_index[move.rsprData.x_clv_index];
  Node *y = ann_network.network.nodes_by_index[move.rsprData.y_clv_index];
  Node *x_prime =
      ann_network.network.nodes_by_index[move.rsprData.x_prime_clv_index];
  Node *y_prime =
      ann_network.network.nodes_by_index[move.rsprData.y_prime_clv_index];
  Node *z = ann_network.network.nodes_by_index[move.rsprData.z_clv_index];
  Edge *x_y_edge = getEdgeTo(ann_network.network, x, y);
  Edge *x_prime_z_edge = getEdgeTo(ann_network.network, x_prime, z);
  Edge *z_y_prime_edge = getEdgeTo(ann_network.network, z, y_prime);
  return {x_y_edge->pmatrix_index, x_prime_z_edge->pmatrix_index,
          z_y_prime_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidatesUndoRSPR(
    AnnotatedNetwork &ann_network, const Move &move) {
  Node *x = ann_network.network.nodes_by_index[move.rsprData.x_clv_index];
  Node *y = ann_network.network.nodes_by_index[move.rsprData.y_clv_index];
  Node *x_prime =
      ann_network.network.nodes_by_index[move.rsprData.x_prime_clv_index];
  Node *y_prime =
      ann_network.network.nodes_by_index[move.rsprData.y_prime_clv_index];
  Node *z = ann_network.network.nodes_by_index[move.rsprData.z_clv_index];
  Edge *x_prime_y_prime_edge = getEdgeTo(ann_network.network, x_prime, y_prime);
  Edge *x_z_edge = getEdgeTo(ann_network.network, x, z);
  Edge *z_y_edge = getEdgeTo(ann_network.network, z, y);
  return {x_prime_y_prime_edge->pmatrix_index, x_z_edge->pmatrix_index,
          z_y_edge->pmatrix_index};
}

Move randomMoveRSPR(AnnotatedNetwork &ann_network) {
  // TODO: This can be made faster
  std::unordered_set<Edge *> tried;
  while (tried.size() != ann_network.network.num_branches()) {
    Edge *edge = getRandomEdge(ann_network);
    if (tried.count(edge) > 0) {
      continue;
    } else {
      tried.emplace(edge);
    }
    auto moves = possibleMovesRSPR(ann_network, edge);
    if (!moves.empty()) {
      return moves[getRandomIndex(ann_network.rng, moves.size())];
    }
  }
  throw std::runtime_error("No random move found");
}

Move randomMoveRSPR1(AnnotatedNetwork &ann_network) {
  // TODO: This can be made faster
  std::unordered_set<Edge *> tried;
  while (tried.size() != ann_network.network.num_branches()) {
    Edge *edge = getRandomEdge(ann_network);
    if (tried.count(edge) > 0) {
      continue;
    } else {
      tried.emplace(edge);
    }
    auto moves = possibleMovesRSPR1(ann_network, edge);
    if (!moves.empty()) {
      return moves[getRandomIndex(ann_network.rng, moves.size())];
    }
  }
  throw std::runtime_error("No random move found");
}

Move randomMoveTail(AnnotatedNetwork &ann_network) {
  // TODO: This can be made faster
  std::unordered_set<Edge *> tried;
  while (tried.size() != ann_network.network.num_branches()) {
    Edge *edge = getRandomEdge(ann_network);
    if (tried.count(edge) > 0) {
      continue;
    } else {
      tried.emplace(edge);
    }
    auto moves = possibleMovesTail(ann_network, edge);
    if (!moves.empty()) {
      return moves[getRandomIndex(ann_network.rng, moves.size())];
    }
  }
  throw std::runtime_error("No random move found");
}

Move randomMoveHead(AnnotatedNetwork &ann_network) {
  // TODO: This can be made faster
  std::unordered_set<Edge *> tried;
  while (tried.size() != ann_network.network.num_branches()) {
    Edge *edge = getRandomEdge(ann_network);
    if (tried.count(edge) > 0) {
      continue;
    } else {
      tried.emplace(edge);
    }
    auto moves = possibleMovesHead(ann_network, edge);
    if (!moves.empty()) {
      return moves[getRandomIndex(ann_network.rng, moves.size())];
    }
  }
  throw std::runtime_error("No random move found");
}

void updateMoveClvIndexRSPR(Move &move, size_t old_clv_index,
                            size_t new_clv_index, bool undo) {
  if (old_clv_index == new_clv_index) {
    return;
  }
  if (!undo) {
    move.remapped_clv_indices.emplace_back(
        std::make_pair(old_clv_index, new_clv_index));
  }

  if (move.rsprData.x_clv_index == old_clv_index) {
    move.rsprData.x_clv_index = new_clv_index;
  } else if (move.rsprData.x_clv_index == new_clv_index) {
    move.rsprData.x_clv_index = old_clv_index;
  }
  if (move.rsprData.y_clv_index == old_clv_index) {
    move.rsprData.y_clv_index = new_clv_index;
  } else if (move.rsprData.y_clv_index == new_clv_index) {
    move.rsprData.y_clv_index = old_clv_index;
  }
  if (move.rsprData.z_clv_index == old_clv_index) {
    move.rsprData.z_clv_index = new_clv_index;
  } else if (move.rsprData.z_clv_index == new_clv_index) {
    move.rsprData.z_clv_index = old_clv_index;
  }
  if (move.rsprData.x_prime_clv_index == old_clv_index) {
    move.rsprData.x_prime_clv_index = new_clv_index;
  } else if (move.rsprData.x_prime_clv_index == new_clv_index) {
    move.rsprData.x_prime_clv_index = old_clv_index;
  }
  if (move.rsprData.y_prime_clv_index == old_clv_index) {
    move.rsprData.y_prime_clv_index = new_clv_index;
  } else if (move.rsprData.y_prime_clv_index == new_clv_index) {
    move.rsprData.y_prime_clv_index = old_clv_index;
  }
  if (move.rsprData.y_prime_first_parent_clv_index == old_clv_index) {
    move.rsprData.y_prime_first_parent_clv_index = new_clv_index;
  } else if (move.rsprData.y_prime_first_parent_clv_index == new_clv_index) {
    move.rsprData.y_prime_first_parent_clv_index = old_clv_index;
  }
  if (move.rsprData.y_first_parent_clv_index == old_clv_index) {
    move.rsprData.y_first_parent_clv_index = new_clv_index;
  } else if (move.rsprData.y_first_parent_clv_index == new_clv_index) {
    move.rsprData.y_first_parent_clv_index = old_clv_index;
  }
  if (move.rsprData.z_first_parent_clv_index == old_clv_index) {
    move.rsprData.z_first_parent_clv_index = new_clv_index;
  } else if (move.rsprData.z_first_parent_clv_index == new_clv_index) {
    move.rsprData.z_first_parent_clv_index = old_clv_index;
  }
}

void updateMovePmatrixIndexRSPR(Move &move, size_t old_pmatrix_index,
                                size_t new_pmatrix_index, bool undo) {
  if (old_pmatrix_index == new_pmatrix_index) {
    return;
  }
  if (!undo) {
    move.remapped_pmatrix_indices.emplace_back(
        std::make_pair(old_pmatrix_index, new_pmatrix_index));
  }
}

}  // namespace netrax