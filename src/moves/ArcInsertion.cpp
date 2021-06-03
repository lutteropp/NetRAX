#include "ArcInsertion.hpp"
#include "ArcInsertionData.hpp"
#include "ArcRemoval.hpp"

#include "../DebugPrintFunctions.hpp"
#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"
#include "GeneralMoveFunctions.hpp"

namespace netrax {

bool checkSanityArcInsertion(AnnotatedNetwork &ann_network, const Move &move) {
  bool good = true;
  good &= (move.moveType == MoveType::ArcInsertionMove ||
           move.moveType == MoveType::DeltaPlusMove);
  good &=
      (ann_network.network.nodes_by_index[move.arcInsertionData.a_clv_index] !=
       nullptr);
  good &=
      (ann_network.network.nodes_by_index[move.arcInsertionData.b_clv_index] !=
       nullptr);
  good &=
      (ann_network.network.nodes_by_index[move.arcInsertionData.c_clv_index] !=
       nullptr);
  good &=
      (ann_network.network.nodes_by_index[move.arcInsertionData.d_clv_index] !=
       nullptr);
  good &=
      (ann_network.network
           .edges_by_index[move.arcInsertionData.ab_pmatrix_index] != nullptr);
  good &=
      (ann_network.network
           .edges_by_index[move.arcInsertionData.cd_pmatrix_index] != nullptr);
  good &=
      (move.arcInsertionData.a_clv_index != move.arcInsertionData.b_clv_index);
  good &=
      (move.arcInsertionData.c_clv_index != move.arcInsertionData.d_clv_index);
  if (good)
    good &= (hasNeighbor(
        ann_network.network.nodes_by_index[move.arcInsertionData.a_clv_index],
        ann_network.network.nodes_by_index[move.arcInsertionData.b_clv_index]));
  if (good)
    good &= (hasNeighbor(
        ann_network.network.nodes_by_index[move.arcInsertionData.c_clv_index],
        ann_network.network.nodes_by_index[move.arcInsertionData.d_clv_index]));
  if (good)
    good &= (!hasPath(
        ann_network.network,
        ann_network.network.nodes_by_index[move.arcInsertionData.d_clv_index],
        ann_network.network.nodes_by_index[move.arcInsertionData.a_clv_index]));
  return good;
}

bool checkSanityArcInsertion(AnnotatedNetwork &ann_network,
                             const std::vector<Move> &moves) {
  bool sane = true;
  for (size_t i = 0; i < moves.size(); ++i) {
    sane &= checkSanityArcInsertion(ann_network, moves[i]);
  }
  return sane;
}

Move buildMoveArcInsertion(
    AnnotatedNetwork &ann_network, size_t a_clv_index, size_t b_clv_index,
    size_t c_clv_index, size_t d_clv_index, std::vector<double> &u_v_len,
    std::vector<double> &c_v_len, std::vector<double> &a_u_len,
    std::vector<double> &a_b_len, std::vector<double> &c_d_len,
    std::vector<double> &v_d_len, std::vector<double> &u_b_len,
    MoveType moveType, size_t edge_orig_idx, size_t node_orig_idx) {
  Move move = Move(moveType, edge_orig_idx, node_orig_idx);
  move.arcInsertionData.a_clv_index = a_clv_index;
  move.arcInsertionData.b_clv_index = b_clv_index;
  move.arcInsertionData.c_clv_index = c_clv_index;
  move.arcInsertionData.d_clv_index = d_clv_index;

  move.arcInsertionData.u_v_len = u_v_len;
  for (size_t i = 0; i < u_v_len.size(); ++i) {
    assert(u_v_len[i] >= ann_network.options.brlen_min &&
           u_v_len[i] <= ann_network.options.brlen_max);
  }
  move.arcInsertionData.c_v_len = c_v_len;
  for (size_t i = 0; i < c_v_len.size(); ++i) {
    assert(c_v_len[i] >= ann_network.options.brlen_min &&
           c_v_len[i] <= ann_network.options.brlen_max);
  }
  move.arcInsertionData.a_u_len = a_u_len;
  for (size_t i = 0; i < a_u_len.size(); ++i) {
    assert(a_u_len[i] >= ann_network.options.brlen_min &&
           a_u_len[i] <= ann_network.options.brlen_max);
  }

  move.arcInsertionData.a_b_len = a_b_len;
  for (size_t i = 0; i < a_b_len.size(); ++i) {
    assert(a_b_len[i] >= ann_network.options.brlen_min &&
           a_b_len[i] <= ann_network.options.brlen_max);
  }
  move.arcInsertionData.c_d_len = c_d_len;
  for (size_t i = 0; i < c_d_len.size(); ++i) {
    assert(c_d_len[i] >= ann_network.options.brlen_min &&
           c_d_len[i] <= ann_network.options.brlen_max);
  }
  move.arcInsertionData.v_d_len = v_d_len;
  for (size_t i = 0; i < v_d_len.size(); ++i) {
    assert(v_d_len[i] >= ann_network.options.brlen_min &&
           v_d_len[i] <= ann_network.options.brlen_max);
  }
  move.arcInsertionData.u_b_len = u_b_len;
  for (size_t i = 0; i < u_b_len.size(); ++i) {
    assert(u_b_len[i] >= ann_network.options.brlen_min &&
           u_b_len[i] <= ann_network.options.brlen_max);
  }

  move.moveType = moveType;
  return move;
}

std::vector<Move> possibleMovesArcInsertion(
    AnnotatedNetwork &ann_network, const Edge *edge, const Node *c,
    const Node *d, MoveType moveType, bool noDeltaPlus, int min_radius = 0,
    int max_radius = std::numeric_limits<int>::max()) {
  size_t edge_orig_idx = edge->pmatrix_index;
  std::vector<Move> res;
  Network &network = ann_network.network;
  // choose two distinct arcs ab, cd (with cd not ancestral to ab -> no d-a-path
  // allowed)
  const Node *a = getSource(network, edge);
  const Node *b = getTarget(network, edge);
  size_t node_orig_idx = b->clv_index;
  std::vector<double> a_b_len =
      get_edge_lengths(ann_network, edge->pmatrix_index);

  std::vector<Node *> radius_nodes =
      getNeighborsWithinRadius(network, b, min_radius, max_radius);

  double min_br = ann_network.options.brlen_min;

  Node *c_cand = nullptr;
  Node *d_cand = nullptr;
  if (c) {
    c_cand = ann_network.network.nodes_by_index[c->clv_index];
    for (size_t i = 0; i < c->links.size(); ++i) {
      if (c->links[i].direction == Direction::INCOMING) {
        continue;
      }
      const Node *d_cand =
          network.nodes_by_index[c->links[i].outer->node_clv_index];
      if (a->clv_index == c_cand->clv_index &&
          b->clv_index == d_cand->clv_index) {
        continue;
      }

      if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) ||
                          (b->clv_index == c_cand->clv_index) ||
                          (b->clv_index == d_cand->clv_index))) {
        continue;
      }

      if (std::find(radius_nodes.begin(), radius_nodes.end(), d_cand) ==
          radius_nodes.end()) {
        continue;
      }

      if (!hasPath(network, d_cand, a)) {
        std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len,
            u_v_len;

        c_d_len = get_edge_lengths(ann_network, c->links[i].edge_pmatrix_index);
        c_v_len = get_halved_edge_lengths(ann_network, c_d_len, min_br);
        a_u_len = get_halved_edge_lengths(ann_network, a_b_len, min_br);
        v_d_len = get_minus_edge_lengths(ann_network, c_d_len, c_v_len, min_br);
        u_b_len = get_minus_edge_lengths(ann_network, a_b_len, a_u_len, min_br);
        u_v_len =
            std::vector<double>(ann_network.fake_treeinfo->brlen_linkage ==
                                        PLLMOD_COMMON_BRLEN_UNLINKED
                                    ? ann_network.fake_treeinfo->partition_count
                                    : 1,
                                1.0);

        Move move = buildMoveArcInsertion(
            ann_network, a->clv_index, b->clv_index, c_cand->clv_index,
            d_cand->clv_index, u_v_len, c_v_len, a_u_len, a_b_len, c_d_len,
            v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);
        move.arcInsertionData.ab_pmatrix_index =
            getEdgeTo(network, a, b)->pmatrix_index;
        move.arcInsertionData.cd_pmatrix_index =
            getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
        res.emplace_back(move);
      }
    }
  } else if (d) {
    d_cand = ann_network.network.nodes_by_index[d->clv_index];
    if (std::find(radius_nodes.begin(), radius_nodes.end(), d) !=
        radius_nodes.end()) {
      if (!hasPath(network, d_cand, a)) {
        for (size_t i = 0; i < d->links.size(); ++i) {
          if (d->links[i].direction == Direction::OUTGOING) {
            continue;
          }
          Node *c_cand =
              network.nodes_by_index[d->links[i].outer->node_clv_index];
          if (a->clv_index == c_cand->clv_index &&
              b->clv_index == d_cand->clv_index) {
            continue;
          }

          if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) ||
                              (b->clv_index == c_cand->clv_index) ||
                              (b->clv_index == d_cand->clv_index))) {
            continue;
          }

          std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len,
              u_v_len;
          c_d_len =
              get_edge_lengths(ann_network, d->links[i].edge_pmatrix_index);
          c_v_len = get_halved_edge_lengths(ann_network, c_d_len, min_br);
          a_u_len = get_halved_edge_lengths(ann_network, a_b_len, min_br);
          v_d_len =
              get_minus_edge_lengths(ann_network, c_d_len, c_v_len, min_br);
          u_b_len =
              get_minus_edge_lengths(ann_network, a_b_len, a_u_len, min_br);
          u_v_len = std::vector<double>(
              ann_network.fake_treeinfo->brlen_linkage ==
                      PLLMOD_COMMON_BRLEN_UNLINKED
                  ? ann_network.fake_treeinfo->partition_count
                  : 1,
              1.0);

          Move move = buildMoveArcInsertion(
              ann_network, a->clv_index, b->clv_index, c_cand->clv_index,
              d_cand->clv_index, u_v_len, c_v_len, a_u_len, a_b_len, c_d_len,
              v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

          move.arcInsertionData.ab_pmatrix_index =
              getEdgeTo(network, a, b)->pmatrix_index;
          move.arcInsertionData.cd_pmatrix_index =
              getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
          res.emplace_back(move);
        }
      }
    }
  } else {
    for (size_t i = 0; i < network.num_branches(); ++i) {
      if (i == edge->pmatrix_index) {
        continue;
      }
      Node *c_cand = getSource(network, network.edges_by_index[i]);
      Node *d_cand = getTarget(network, network.edges_by_index[i]);

      if (std::find(radius_nodes.begin(), radius_nodes.end(), d_cand) ==
          radius_nodes.end()) {
        continue;
      }

      if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) ||
                          (b->clv_index == c_cand->clv_index) ||
                          (b->clv_index == d_cand->clv_index))) {
        continue;
      }

      if (!hasPath(network, d_cand, a)) {
        std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len,
            u_v_len;
        c_d_len = get_edge_lengths(ann_network, i);
        c_v_len = get_halved_edge_lengths(ann_network, c_d_len, min_br);
        a_u_len = get_halved_edge_lengths(ann_network, a_b_len, min_br);
        v_d_len = get_minus_edge_lengths(ann_network, c_d_len, c_v_len, min_br);
        u_b_len = get_minus_edge_lengths(ann_network, a_b_len, a_u_len, min_br);
        u_v_len =
            std::vector<double>(ann_network.fake_treeinfo->brlen_linkage ==
                                        PLLMOD_COMMON_BRLEN_UNLINKED
                                    ? ann_network.fake_treeinfo->partition_count
                                    : 1,
                                1.0);

        Move move = buildMoveArcInsertion(
            ann_network, a->clv_index, b->clv_index, c_cand->clv_index,
            d_cand->clv_index, u_v_len, c_v_len, a_u_len, a_b_len, c_d_len,
            v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

        move.arcInsertionData.ab_pmatrix_index =
            getEdgeTo(network, a, b)->pmatrix_index;
        move.arcInsertionData.cd_pmatrix_index =
            getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
        res.emplace_back(move);
      }
    }
  }
  return res;
}

std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
                                            const Node *node, const Node *c,
                                            const Node *d, MoveType moveType,
                                            bool noDeltaPlus, int min_radius,
                                            int max_radius) {
  std::vector<Move> res;
  Network &network = ann_network.network;
  if (node == ann_network.network.root) {
    return res;  // because we need a parent
  }

  size_t node_orig_idx = node->clv_index;

  std::vector<Node *> radius_nodes =
      getNeighborsWithinRadius(network, node, min_radius, max_radius);

  const Node *b = node;
  std::vector<Node *> parents;
  if (node->getType() == NodeType::BASIC_NODE) {
    parents.emplace_back(getActiveParent(network, node));
  } else {
    parents.emplace_back(getReticulationFirstParent(network, node));
    parents.emplace_back(getReticulationSecondParent(network, node));
  }
  for (const Node *a : parents) {
    const Edge *edge = getEdgeTo(network, a, b);
    size_t edge_orig_idx = edge->pmatrix_index;
    // choose two distinct arcs ab, cd (with cd not ancestral to ab -> no
    // d-a-path allowed)
    std::vector<double> a_b_len =
        get_edge_lengths(ann_network, edge->pmatrix_index);

    double min_br = ann_network.options.brlen_min;

    const Node *c_cand = nullptr;
    const Node *d_cand = nullptr;
    if (c) {
      c_cand = c;
      for (size_t i = 0; i < c->links.size(); ++i) {
        if (c->links[i].direction == Direction::INCOMING) {
          continue;
        }
        Node *d_cand =
            network.nodes_by_index[c->links[i].outer->node_clv_index];
        if (std::find(radius_nodes.begin(), radius_nodes.end(), d_cand) !=
            radius_nodes.end()) {
          if (a->clv_index == c_cand->clv_index &&
              b->clv_index == d_cand->clv_index) {
            continue;
          }

          if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) ||
                              (b->clv_index == c_cand->clv_index) ||
                              (b->clv_index == d_cand->clv_index))) {
            continue;
          }

          if (!hasPath(network, d_cand, a)) {
            std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len,
                u_v_len;

            c_d_len =
                get_edge_lengths(ann_network, c->links[i].edge_pmatrix_index);
            c_v_len = get_halved_edge_lengths(ann_network, c_d_len, min_br);
            a_u_len = get_halved_edge_lengths(ann_network, a_b_len, min_br);
            v_d_len =
                get_minus_edge_lengths(ann_network, c_d_len, c_v_len, min_br);
            u_b_len =
                get_minus_edge_lengths(ann_network, a_b_len, a_u_len, min_br);
            u_v_len = std::vector<double>(
                ann_network.fake_treeinfo->brlen_linkage ==
                        PLLMOD_COMMON_BRLEN_UNLINKED
                    ? ann_network.fake_treeinfo->partition_count
                    : 1,
                min_br);

            Move move = buildMoveArcInsertion(
                ann_network, a->clv_index, b->clv_index, c_cand->clv_index,
                d_cand->clv_index, u_v_len, c_v_len, a_u_len, a_b_len, c_d_len,
                v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);
            move.arcInsertionData.ab_pmatrix_index =
                getEdgeTo(network, a, b)->pmatrix_index;
            move.arcInsertionData.cd_pmatrix_index =
                getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
            res.emplace_back(move);
          }
        }
      }
    } else if (d) {
      d_cand = d;

      if (std::find(radius_nodes.begin(), radius_nodes.end(), d_cand) !=
          radius_nodes.end()) {
        if (!hasPath(network, d_cand, a)) {
          for (size_t i = 0; i < d->links.size(); ++i) {
            if (d->links[i].direction == Direction::OUTGOING) {
              continue;
            }
            Node *c_cand =
                network.nodes_by_index[d->links[i].outer->node_clv_index];
            if (a->clv_index == c_cand->clv_index &&
                b->clv_index == d_cand->clv_index) {
              continue;
            }

            if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) ||
                                (b->clv_index == c_cand->clv_index) ||
                                (b->clv_index == d_cand->clv_index))) {
              continue;
            }

            std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len,
                u_v_len;
            c_d_len =
                get_edge_lengths(ann_network, d->links[i].edge_pmatrix_index);
            c_v_len = get_halved_edge_lengths(ann_network, c_d_len, min_br);
            a_u_len = get_halved_edge_lengths(ann_network, a_b_len, min_br);
            v_d_len =
                get_minus_edge_lengths(ann_network, c_d_len, c_v_len, min_br);
            u_b_len =
                get_minus_edge_lengths(ann_network, a_b_len, a_u_len, min_br);
            u_v_len = std::vector<double>(
                ann_network.fake_treeinfo->brlen_linkage ==
                        PLLMOD_COMMON_BRLEN_UNLINKED
                    ? ann_network.fake_treeinfo->partition_count
                    : 1,
                1.0);

            Move move = buildMoveArcInsertion(
                ann_network, a->clv_index, b->clv_index, c_cand->clv_index,
                d_cand->clv_index, u_v_len, c_v_len, a_u_len, a_b_len, c_d_len,
                v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

            move.arcInsertionData.ab_pmatrix_index =
                getEdgeTo(network, a, b)->pmatrix_index;
            move.arcInsertionData.cd_pmatrix_index =
                getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
            res.emplace_back(move);
          }
        }
      }
    } else {
      std::vector<Node *> d_candidates = radius_nodes;
      for (const Node *d_cand : d_candidates) {
        std::vector<Node *> c_candidates = getAllParents(network, d_cand);
        for (const Node *c_cand : c_candidates) {
          const Edge *actEdge = getEdgeTo(network, c_cand, d_cand);
          if (actEdge->pmatrix_index == edge->pmatrix_index) {
            continue;
          }
          if (noDeltaPlus && ((a->clv_index == c_cand->clv_index) ||
                              (b->clv_index == c_cand->clv_index) ||
                              (b->clv_index == d_cand->clv_index))) {
            continue;
          }

          if (!hasPath(network, d_cand, a)) {
            std::vector<double> c_d_len, c_v_len, a_u_len, v_d_len, u_b_len,
                u_v_len;
            c_d_len = get_edge_lengths(ann_network, actEdge->pmatrix_index);
            c_v_len = get_halved_edge_lengths(ann_network, c_d_len, min_br);
            a_u_len = get_halved_edge_lengths(ann_network, a_b_len, min_br);
            v_d_len =
                get_minus_edge_lengths(ann_network, c_d_len, c_v_len, min_br);
            u_b_len =
                get_minus_edge_lengths(ann_network, a_b_len, a_u_len, min_br);
            u_v_len = std::vector<double>(
                ann_network.fake_treeinfo->brlen_linkage ==
                        PLLMOD_COMMON_BRLEN_UNLINKED
                    ? ann_network.fake_treeinfo->partition_count
                    : 1,
                1.0);

            Move move = buildMoveArcInsertion(
                ann_network, a->clv_index, b->clv_index, c_cand->clv_index,
                d_cand->clv_index, u_v_len, c_v_len, a_u_len, a_b_len, c_d_len,
                v_d_len, u_b_len, moveType, edge_orig_idx, node_orig_idx);

            move.arcInsertionData.ab_pmatrix_index =
                getEdgeTo(network, a, b)->pmatrix_index;
            move.arcInsertionData.cd_pmatrix_index =
                getEdgeTo(network, c_cand, d_cand)->pmatrix_index;
            res.emplace_back(move);
          }
        }
      }
    }
  }
  return res;
}

std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
                                            const Edge *edge, bool noDeltaPlus,
                                            int min_radius, int max_radius) {
  assert(edge);
  return possibleMovesArcInsertion(ann_network, edge, nullptr, nullptr,
                                   MoveType::ArcInsertionMove, noDeltaPlus,
                                   min_radius, max_radius);
}

std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
                                            const Node *node, bool noDeltaPlus,
                                            int min_radius, int max_radius) {
  assert(node);
  return possibleMovesArcInsertion(ann_network, node, nullptr, nullptr,
                                   MoveType::ArcInsertionMove, noDeltaPlus,
                                   min_radius, max_radius);
}

std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
                                         const Edge *edge, int min_radius,
                                         int max_radius) {
  assert(edge);
  Network &network = ann_network.network;
  std::vector<Move> res;
  Node *a = getSource(network, edge);
  Node *b = getTarget(network, edge);

  // Case 1: a == c
  std::vector<Move> case1 = possibleMovesArcInsertion(
      ann_network, edge, a, nullptr, MoveType::DeltaPlusMove, false, min_radius,
      max_radius);
  res.insert(std::end(res), std::begin(case1), std::end(case1));

  // Case 2: b == d
  std::vector<Move> case2 = possibleMovesArcInsertion(
      ann_network, edge, nullptr, b, MoveType::DeltaPlusMove, false, min_radius,
      max_radius);
  res.insert(std::end(res), std::begin(case2), std::end(case2));

  // Case 3: b == c
  std::vector<Move> case3 = possibleMovesArcInsertion(
      ann_network, edge, b, nullptr, MoveType::DeltaPlusMove, false, min_radius,
      max_radius);
  res.insert(std::end(res), std::begin(case3), std::end(case3));

  return res;
}

std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
                                         const Node *node, int min_radius,
                                         int max_radius) {
  assert(node);
  Network &network = ann_network.network;
  std::vector<Move> res;
  if (node == ann_network.network.root) {
    return res;  // because we need a parent, and the root has no parent
  }
  const Node *b = node;
  std::vector<Node *> parents = getAllParents(network, node);
  for (const Node *a : parents) {
    // Case 1: a == c
    std::vector<Move> case1 = possibleMovesArcInsertion(
        ann_network, node, a, nullptr, MoveType::DeltaPlusMove, false,
        min_radius, max_radius);
    res.insert(std::end(res), std::begin(case1), std::end(case1));

    if (min_radius == 0) {
      // Case 2: b == d
      std::vector<Move> case2 = possibleMovesArcInsertion(
          ann_network, node, nullptr, b, MoveType::DeltaPlusMove, false,
          min_radius, max_radius);
      res.insert(std::end(res), std::begin(case2), std::end(case2));
    }

    // Case 3: b == c
    std::vector<Move> case3 = possibleMovesArcInsertion(
        ann_network, node, b, nullptr, MoveType::DeltaPlusMove, false,
        min_radius, max_radius);
    res.insert(std::end(res), std::begin(case3), std::end(case3));
  }
  return res;
}

std::vector<Move> possibleMovesArcInsertion(
    AnnotatedNetwork &ann_network, const std::vector<Node *> &start_nodes,
    bool noDeltaPlus, int min_radius, int max_radius) {
  std::vector<Move> res;
  for (const Node *node : start_nodes) {
    assert(node);
    std::vector<Move> res_node = possibleMovesArcInsertion(
        ann_network, node, noDeltaPlus, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_node), std::end(res_node));
  }
  return res;
}

std::vector<Move> possibleMovesArcInsertion(
    AnnotatedNetwork &ann_network, const std::vector<Edge *> &start_edges,
    bool noDeltaPlus, int min_radius, int max_radius) {
  std::vector<Move> res;
  for (const Edge *edge : start_edges) {
    assert(edge);
    std::vector<Move> res_edge = possibleMovesArcInsertion(
        ann_network, edge, noDeltaPlus, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_edge), std::end(res_edge));
  }
  return res;
}

std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
                                         const std::vector<Node *> &start_nodes,
                                         int min_radius, int max_radius) {
  std::vector<Move> res;
  for (Node *node : start_nodes) {
    assert(node);
    std::vector<Move> res_node =
        possibleMovesDeltaPlus(ann_network, node, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_node), std::end(res_node));
  }
  return res;
}

std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
                                         const std::vector<Edge *> &start_edges,
                                         int min_radius, int max_radius) {
  std::vector<Move> res;
  for (Edge *edge : start_edges) {
    assert(edge);
    std::vector<Move> res_edge =
        possibleMovesDeltaPlus(ann_network, edge, min_radius, max_radius);
    res.insert(std::end(res), std::begin(res_edge), std::end(res_edge));
  }
  return res;
}

std::vector<Move> possibleMovesArcInsertion(AnnotatedNetwork &ann_network,
                                            bool noDeltaPlus, int min_radius,
                                            int max_radius) {
  std::vector<Move> res;
  Network &network = ann_network.network;
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    std::vector<Move> node_moves = possibleMovesArcInsertion(
        ann_network, network.nodes_by_index[i], nullptr, nullptr,
        MoveType::ArcInsertionMove, noDeltaPlus, min_radius, max_radius);
    res.insert(std::end(res), std::begin(node_moves), std::end(node_moves));
  }
  sortByProximity(res, ann_network);
  assert(checkSanityArcInsertion(ann_network, res));
  return res;
}

std::vector<Move> possibleMovesDeltaPlus(AnnotatedNetwork &ann_network,
                                         int min_radius, int max_radius) {
  std::vector<Move> res;
  Network &network = ann_network.network;
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    std::vector<Move> node_moves = possibleMovesDeltaPlus(
        ann_network, network.nodes_by_index[i], min_radius, max_radius);
    res.insert(std::end(res), std::begin(node_moves), std::end(node_moves));
  }
  sortByProximity(res, ann_network);
  assert(checkSanityArcInsertion(ann_network, res));
  return res;
}

void updateMoveClvIndexArcInsertion(Move &move, size_t old_clv_index,
                                    size_t new_clv_index, bool undo) {
  if (old_clv_index == new_clv_index) {
    return;
  }
  if (!undo) {
    move.remapped_clv_indices.emplace_back(
        std::make_pair(old_clv_index, new_clv_index));
  }
  if (move.arcInsertionData.a_clv_index == old_clv_index) {
    move.arcInsertionData.a_clv_index = new_clv_index;
  } else if (move.arcInsertionData.a_clv_index == new_clv_index) {
    move.arcInsertionData.a_clv_index = old_clv_index;
  }
  if (move.arcInsertionData.b_clv_index == old_clv_index) {
    move.arcInsertionData.b_clv_index = new_clv_index;
  } else if (move.arcInsertionData.b_clv_index == new_clv_index) {
    move.arcInsertionData.b_clv_index = old_clv_index;
  }
  if (move.arcInsertionData.c_clv_index == old_clv_index) {
    move.arcInsertionData.c_clv_index = new_clv_index;
  } else if (move.arcInsertionData.c_clv_index == new_clv_index) {
    move.arcInsertionData.c_clv_index = old_clv_index;
  }
  if (move.arcInsertionData.d_clv_index == old_clv_index) {
    move.arcInsertionData.d_clv_index = new_clv_index;
  } else if (move.arcInsertionData.d_clv_index == new_clv_index) {
    move.arcInsertionData.d_clv_index = old_clv_index;
  }
}

void updateMovePmatrixIndexArcInsertion(Move &move, size_t old_pmatrix_index,
                                        size_t new_pmatrix_index, bool undo) {
  if (old_pmatrix_index == new_pmatrix_index) {
    return;
  }
  if (!undo) {
    move.remapped_pmatrix_indices.emplace_back(
        std::make_pair(old_pmatrix_index, new_pmatrix_index));
  }
  if (move.arcInsertionData.ab_pmatrix_index == old_pmatrix_index) {
    move.arcInsertionData.ab_pmatrix_index = new_pmatrix_index;
  } else if (move.arcInsertionData.ab_pmatrix_index == new_pmatrix_index) {
    move.arcInsertionData.ab_pmatrix_index = old_pmatrix_index;
  }
  if (move.arcInsertionData.cd_pmatrix_index == old_pmatrix_index) {
    move.arcInsertionData.cd_pmatrix_index = new_pmatrix_index;
  } else if (move.arcInsertionData.cd_pmatrix_index == new_pmatrix_index) {
    move.arcInsertionData.cd_pmatrix_index = old_pmatrix_index;
  }
}

void performMoveArcInsertion(AnnotatedNetwork &ann_network, Move &move) {
  assert(checkSanityArcInsertion(ann_network, move));
  assert(move.moveType == MoveType::ArcInsertionMove ||
         move.moveType == MoveType::DeltaPlusMove);
  Network &network = ann_network.network;
  checkSanity(ann_network);

  Link *from_a_link = getLinkToNode(network, move.arcInsertionData.a_clv_index,
                                    move.arcInsertionData.b_clv_index);
  Link *to_b_link = getLinkToNode(network, move.arcInsertionData.b_clv_index,
                                  move.arcInsertionData.a_clv_index);
  Link *from_c_link = getLinkToNode(network, move.arcInsertionData.c_clv_index,
                                    move.arcInsertionData.d_clv_index);
  Link *to_d_link = getLinkToNode(network, move.arcInsertionData.d_clv_index,
                                  move.arcInsertionData.c_clv_index);
  Edge *a_b_edge = getEdgeTo(network, move.arcInsertionData.a_clv_index,
                             move.arcInsertionData.b_clv_index);
  if (move.arcInsertionData.ab_pmatrix_index != a_b_edge->pmatrix_index) {
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << exportDebugInfo(ann_network);
      std::cout << "move.arcInsertionData.ab_pmatrix_index: "
                << move.arcInsertionData.ab_pmatrix_index << "\n";
      std::cout << "a_b_edge->pmatrix_index: " << a_b_edge->pmatrix_index
                << "\n";
    }
    throw std::runtime_error("invalid move data");
  }
  assert(a_b_edge->link1);
  assert(a_b_edge->link2);
  Edge *c_d_edge = getEdgeTo(network, move.arcInsertionData.c_clv_index,
                             move.arcInsertionData.d_clv_index);
  if (move.arcInsertionData.cd_pmatrix_index != c_d_edge->pmatrix_index) {
    std::cout << exportDebugInfo(ann_network);
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
      std::cout << "move.arcInsertionData.cd_pmatrix_index: "
                << move.arcInsertionData.cd_pmatrix_index << "\n";
      std::cout << "c_d_edge->pmatrix_index: " << c_d_edge->pmatrix_index
                << "\n";
    }
    throw std::runtime_error("invalid move data");
  }
  assert(c_d_edge->link1);
  assert(c_d_edge->link2);

  ReticulationData retData;
  retData.init(network.num_reticulations(), "", 0, nullptr, nullptr, nullptr);
  Node *u = addInnerNode(ann_network, nullptr,
                         move.arcInsertionData.wanted_u_clv_index);
  Node *v = addInnerNode(ann_network, &retData,
                         move.arcInsertionData.wanted_v_clv_index);

  move.arcInsertionData.wanted_u_clv_index = u->clv_index;
  move.arcInsertionData.wanted_v_clv_index = v->clv_index;

  Link *to_u_link = make_link(u, nullptr, Direction::INCOMING);
  Link *u_b_link = make_link(u, nullptr, Direction::OUTGOING);
  Link *u_v_link = make_link(u, nullptr, Direction::OUTGOING);

  Link *v_u_link = make_link(v, nullptr, Direction::INCOMING);
  Link *v_c_link = make_link(v, nullptr, Direction::INCOMING);
  Link *v_d_link = make_link(v, nullptr, Direction::OUTGOING);

  std::vector<double> u_v_edge_length = move.arcInsertionData.u_v_len;
  for (size_t i = 0; i < u_v_edge_length.size(); ++i) {
    assert(u_v_edge_length[i] >= ann_network.options.brlen_min &&
           u_v_edge_length[i] <= ann_network.options.brlen_max);
  }
  std::vector<double> c_v_edge_length = move.arcInsertionData.c_v_len;
  for (size_t i = 0; i < c_v_edge_length.size(); ++i) {
    assert(c_v_edge_length[i] >= ann_network.options.brlen_min &&
           c_v_edge_length[i] <= ann_network.options.brlen_max);
  }
  std::vector<double> v_d_edge_length = move.arcInsertionData.v_d_len;
  for (size_t i = 0; i < v_d_edge_length.size(); ++i) {
    assert(v_d_edge_length[i] >= ann_network.options.brlen_min &&
           v_d_edge_length[i] <= ann_network.options.brlen_max);
  }
  std::vector<double> a_u_edge_length = move.arcInsertionData.a_u_len;
  for (size_t i = 0; i < a_u_edge_length.size(); ++i) {
    assert(a_u_edge_length[i] >= ann_network.options.brlen_min &&
           a_u_edge_length[i] <= ann_network.options.brlen_max);
  }
  std::vector<double> u_b_edge_length = move.arcInsertionData.u_b_len;
  for (size_t i = 0; i < u_b_edge_length.size(); ++i) {
    assert(u_b_edge_length[i] >= ann_network.options.brlen_min &&
           u_b_edge_length[i] <= ann_network.options.brlen_max);
  }

  removeEdge(ann_network, move,
             network.edges_by_index[move.arcInsertionData.ab_pmatrix_index],
             false);
  if (move.arcInsertionData.cd_pmatrix_index !=
      move.arcInsertionData.ab_pmatrix_index) {
    removeEdge(ann_network, move,
               network.edges_by_index[move.arcInsertionData.cd_pmatrix_index],
               false);
  }

  Edge *u_b_edge = addEdge(ann_network, u_b_link, to_b_link, u_b_edge_length[0],
                           move.arcInsertionData.wanted_ub_pmatrix_index);
  Edge *a_u_edge =
      addEdge(ann_network, from_a_link, to_u_link, a_u_edge_length[0],
              move.arcInsertionData.wanted_au_pmatrix_index);
  Edge *c_v_edge =
      addEdge(ann_network, from_c_link, v_c_link, c_v_edge_length[0],
              move.arcInsertionData.wanted_cv_pmatrix_index);
  Edge *u_v_edge = addEdge(ann_network, u_v_link, v_u_link, u_v_edge_length[0],
                           move.arcInsertionData.wanted_uv_pmatrix_index);
  Edge *v_d_edge = addEdge(ann_network, v_d_link, to_d_link, v_d_edge_length[0],
                           move.arcInsertionData.wanted_vd_pmatrix_index);

  move.arcInsertionData.wanted_au_pmatrix_index = a_u_edge->pmatrix_index;
  move.arcInsertionData.wanted_cv_pmatrix_index = c_v_edge->pmatrix_index;
  move.arcInsertionData.wanted_ub_pmatrix_index = u_b_edge->pmatrix_index;
  move.arcInsertionData.wanted_vd_pmatrix_index = v_d_edge->pmatrix_index;
  move.arcInsertionData.wanted_uv_pmatrix_index = u_v_edge->pmatrix_index;

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

  v->getReticulationData()->link_to_first_parent = v_u_link;
  v->getReticulationData()->link_to_second_parent = v_c_link;
  v->getReticulationData()->link_to_child = v_d_link;
  if (v->getReticulationData()->link_to_first_parent->outer->node_clv_index >
      v->getReticulationData()->link_to_second_parent->outer->node_clv_index) {
    std::swap(v->getReticulationData()->link_to_first_parent,
              v->getReticulationData()->link_to_second_parent);
  }

  set_edge_lengths(ann_network, u_b_edge->pmatrix_index, u_b_edge_length);
  set_edge_lengths(ann_network, v_d_edge->pmatrix_index, v_d_edge_length);
  set_edge_lengths(ann_network, a_u_edge->pmatrix_index, a_u_edge_length);
  set_edge_lengths(ann_network, c_v_edge->pmatrix_index, c_v_edge_length);
  set_edge_lengths(ann_network, u_v_edge->pmatrix_index, u_v_edge_length);

  std::vector<size_t> updateMe = {
      u_v_edge->pmatrix_index, c_v_edge->pmatrix_index, v_d_edge->pmatrix_index,
      a_u_edge->pmatrix_index, u_b_edge->pmatrix_index};
  invalidate_pmatrices(ann_network, updateMe);

  std::vector<bool> visited(network.nodes.size(), false);
  invalidateHigherCLVs(
      ann_network, network.nodes_by_index[move.arcInsertionData.a_clv_index],
      false, visited);
  invalidateHigherCLVs(
      ann_network, network.nodes_by_index[move.arcInsertionData.b_clv_index],
      false, visited);
  invalidateHigherCLVs(
      ann_network, network.nodes_by_index[move.arcInsertionData.c_clv_index],
      false, visited);
  invalidateHigherCLVs(
      ann_network, network.nodes_by_index[move.arcInsertionData.d_clv_index],
      false, visited);
  invalidateHigherCLVs(ann_network, u, false, visited);
  invalidateHigherCLVs(ann_network, v, false, visited);
  invalidatePmatrixIndex(ann_network, u_b_edge->pmatrix_index, visited);
  invalidatePmatrixIndex(ann_network, v_d_edge->pmatrix_index, visited);
  invalidatePmatrixIndex(ann_network, a_u_edge->pmatrix_index, visited);
  invalidatePmatrixIndex(ann_network, c_v_edge->pmatrix_index, visited);
  invalidatePmatrixIndex(ann_network, u_v_edge->pmatrix_index, visited);

  fixReticulationLinks(ann_network, move);

  ann_network.travbuffer = reversed_topological_sort(ann_network.network);
  checkSanity(ann_network);
  assert(assertReticulationProbs(ann_network));
  assert(assertConsecutiveIndices(ann_network));
  assert(assertBranchLengths(ann_network));
}

void undoMoveArcInsertion(AnnotatedNetwork &ann_network, Move &move) {
  assert(move.moveType == MoveType::ArcInsertionMove ||
         move.moveType == MoveType::DeltaPlusMove);
  assert(assertConsecutiveIndices(ann_network));
  assert(assertBranchLengths(ann_network));
  Network &network = ann_network.network;
  checkSanity(ann_network);
  const Node *a = network.nodes_by_index[move.arcInsertionData.a_clv_index];
  const Node *b = network.nodes_by_index[move.arcInsertionData.b_clv_index];
  const Node *c = network.nodes_by_index[move.arcInsertionData.c_clv_index];
  const Node *d = network.nodes_by_index[move.arcInsertionData.d_clv_index];

  const Node *u = nullptr;
  const Node *v = nullptr;
  // Find u and v
  std::vector<Node *> uCandidates = getChildren(network, a);
  std::vector<Node *> vCandidates = getChildren(network, c);
  for (size_t i = 0; i < uCandidates.size(); ++i) {
    if (!hasChild(network, uCandidates[i], b)) {
      continue;
    }
    for (size_t j = 0; j < vCandidates.size(); ++j) {
      if (hasChild(network, uCandidates[i], b) &&
          hasChild(network, uCandidates[i], vCandidates[j]) &&
          hasChild(network, vCandidates[j], d)) {
        const Node *u_cand = uCandidates[i];
        const Node *v_cand = vCandidates[j];
        if (u_cand != a && u_cand != b && u_cand != c && u_cand != d &&
            v_cand != a && v_cand != b && v_cand != c && v_cand != d &&
            u_cand != v_cand) {
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
  Move removal = buildMoveArcRemoval(
      ann_network, move.arcInsertionData.a_clv_index,
      move.arcInsertionData.b_clv_index, move.arcInsertionData.c_clv_index,
      move.arcInsertionData.d_clv_index, u->clv_index, v->clv_index,
      move.arcInsertionData.u_v_len, move.arcInsertionData.c_v_len,
      move.arcInsertionData.a_u_len, move.arcInsertionData.a_b_len,
      move.arcInsertionData.c_d_len, move.arcInsertionData.v_d_len,
      move.arcInsertionData.u_b_len, MoveType::ArcRemovalMove,
      move.edge_orig_idx, move.node_orig_idx);
  removal.arcRemovalData.wanted_ab_pmatrix_index =
      move.arcInsertionData.ab_pmatrix_index;
  removal.arcRemovalData.wanted_cd_pmatrix_index =
      move.arcInsertionData.cd_pmatrix_index;
  removal.arcRemovalData.au_pmatrix_index =
      getEdgeTo(network, a, u)->pmatrix_index;
  removal.arcRemovalData.cv_pmatrix_index =
      getEdgeTo(network, c, v)->pmatrix_index;
  removal.arcRemovalData.uv_pmatrix_index =
      getEdgeTo(network, u, v)->pmatrix_index;
  removal.arcRemovalData.ub_pmatrix_index =
      getEdgeTo(network, u, b)->pmatrix_index;
  removal.arcRemovalData.vd_pmatrix_index =
      getEdgeTo(network, v, d)->pmatrix_index;

  performMoveArcRemoval(ann_network, removal);

  // undo all index swaps that have taken place
  for (int i = move.remapped_reticulation_indices.size() - 1; i >= 0; i--) {
    swapReticulationIndex(ann_network, move,
                          move.remapped_reticulation_indices[i].first,
                          move.remapped_reticulation_indices[i].second, true);
  }
  for (int i = move.remapped_clv_indices.size() - 1; i >= 0; i--) {
    swapClvIndex(ann_network, move, move.remapped_clv_indices[i].first,
                 move.remapped_clv_indices[i].second, true);
  }
  for (int i = move.remapped_pmatrix_indices.size() - 1; i >= 0; i--) {
    swapPmatrixIndex(ann_network, move, move.remapped_pmatrix_indices[i].first,
                     move.remapped_pmatrix_indices[i].second, true);
  }

  fixReticulationLinks(ann_network, move);
  assert(assertConsecutiveIndices(ann_network));
  assert(assertBranchLengths(ann_network));
}

std::string toStringArcInsertion(const Move &move) {
  std::stringstream ss;
  ss << "arc insertion move:\n";
  ss << "  a = " << move.arcInsertionData.a_clv_index << "\n";
  ss << "  b = " << move.arcInsertionData.b_clv_index << "\n";
  ss << "  c = " << move.arcInsertionData.c_clv_index << "\n";
  ss << "  d = " << move.arcInsertionData.d_clv_index << "\n";
  ss << "  wanted u = " << move.arcInsertionData.wanted_u_clv_index << "\n";
  ss << "  wanted v = " << move.arcInsertionData.wanted_v_clv_index << "\n";
  ss << "  ab = " << move.arcInsertionData.ab_pmatrix_index << "\n";
  ss << "   a_b_len: " << move.arcInsertionData.a_b_len << "\n";
  ss << "  cd = " << move.arcInsertionData.cd_pmatrix_index << "\n";
  ss << "   c_d_len: " << move.arcInsertionData.c_d_len << "\n";
  ss << "  wanted au = " << move.arcInsertionData.wanted_au_pmatrix_index
     << "\n";
  ss << "   a_u_len: " << move.arcInsertionData.a_u_len << "\n";
  ss << "  wanted cv = " << move.arcInsertionData.wanted_cv_pmatrix_index
     << "\n";
  ss << "   c_v_len: " << move.arcInsertionData.c_v_len << "\n";
  ss << "  wanted ub = " << move.arcInsertionData.wanted_ub_pmatrix_index
     << "\n";
  ss << "   u_b_len: " << move.arcInsertionData.u_b_len << "\n";
  ss << "  wanted vd = " << move.arcInsertionData.wanted_vd_pmatrix_index
     << "\n";
  ss << "   v_d_len: " << move.arcInsertionData.v_d_len << "\n";
  ss << "  wanted uv = " << move.arcInsertionData.wanted_uv_pmatrix_index
     << "\n";
  ss << "   u_v_len: " << move.arcInsertionData.u_v_len << "\n";
  ss << "  remapped_clv_indices: " << move.remapped_clv_indices << "\n";
  ss << "  remapped_pmatrix_indices: " << move.remapped_pmatrix_indices << "\n";
  ss << "  remapped_reticulation_indices: "
     << move.remapped_reticulation_indices << "\n";
  return ss.str();
}

std::unordered_set<size_t> brlenOptCandidatesArcInsertion(
    AnnotatedNetwork &ann_network, const Move &move) {
  Network &network = ann_network.network;
  const Node *a = network.nodes_by_index[move.arcInsertionData.a_clv_index];
  const Node *b = network.nodes_by_index[move.arcInsertionData.b_clv_index];
  const Node *c = network.nodes_by_index[move.arcInsertionData.c_clv_index];
  const Node *d = network.nodes_by_index[move.arcInsertionData.d_clv_index];

  // find u and v
  const Node *u = nullptr;
  const Node *v = nullptr;
  std::vector<Node *> uCandidates = getChildren(network, a);
  std::vector<Node *> vCandidates = getChildren(network, c);
  for (size_t i = 0; i < uCandidates.size(); ++i) {
    if (!hasChild(network, uCandidates[i], b)) {
      continue;
    }
    for (size_t j = 0; j < vCandidates.size(); ++j) {
      if (hasChild(network, uCandidates[i], b) &&
          hasChild(network, uCandidates[i], vCandidates[j]) &&
          hasChild(network, vCandidates[j], d)) {
        const Node *u_cand = uCandidates[i];
        const Node *v_cand = vCandidates[j];
        if (u_cand != a && u_cand != b && u_cand != c && u_cand != d &&
            v_cand != a && v_cand != b && v_cand != c && v_cand != d &&
            u_cand != v_cand) {
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

  const Edge *a_u_edge = getEdgeTo(network, a, u);
  const Edge *u_b_edge = getEdgeTo(network, u, b);
  const Edge *c_v_edge = getEdgeTo(network, c, v);
  const Edge *v_d_edge = getEdgeTo(network, v, d);
  const Edge *u_v_edge = getEdgeTo(network, u, v);
  return {a_u_edge->pmatrix_index, u_b_edge->pmatrix_index,
          c_v_edge->pmatrix_index, v_d_edge->pmatrix_index,
          u_v_edge->pmatrix_index};
}

std::unordered_set<size_t> brlenOptCandidatesUndoArcInsertion(
    AnnotatedNetwork &ann_network, const Move &move) {
  Node *a =
      ann_network.network.nodes_by_index[move.arcInsertionData.a_clv_index];
  Node *b =
      ann_network.network.nodes_by_index[move.arcInsertionData.b_clv_index];
  Node *c =
      ann_network.network.nodes_by_index[move.arcInsertionData.c_clv_index];
  Node *d =
      ann_network.network.nodes_by_index[move.arcInsertionData.d_clv_index];
  Edge *a_b_edge = getEdgeTo(ann_network.network, a, b);
  Edge *c_d_edge = getEdgeTo(ann_network.network, c, d);
  return {a_b_edge->pmatrix_index, c_d_edge->pmatrix_index};
}

Move randomMoveArcInsertion(AnnotatedNetwork &ann_network) {
  // TODO: This can be made faster
  std::unordered_set<Edge *> tried;
  while (tried.size() != ann_network.network.num_branches()) {
    Edge *edge = getRandomEdge(ann_network);
    if (tried.count(edge) > 0) {
      continue;
    } else {
      tried.emplace(edge);
    }
    auto moves = possibleMovesArcInsertion(ann_network, edge);
    if (!moves.empty()) {
      return moves[getRandomIndex(ann_network.rng, moves.size())];
    }
  }
  throw std::runtime_error("No random move found");
}

Move randomMoveDeltaPlus(AnnotatedNetwork &ann_network) {
  // TODO: This can be made faster
  std::unordered_set<Edge *> tried;
  while (tried.size() != ann_network.network.num_branches()) {
    Edge *edge = getRandomEdge(ann_network);
    if (tried.count(edge) > 0) {
      continue;
    } else {
      tried.emplace(edge);
    }
    auto moves = possibleMovesDeltaPlus(ann_network, edge);
    if (!moves.empty()) {
      return moves[getRandomIndex(ann_network.rng, moves.size())];
    }
  }
  throw std::runtime_error("No random move found");
}

}  // namespace netrax