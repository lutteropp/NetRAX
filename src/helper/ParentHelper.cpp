#include "Helper.hpp"

namespace netrax {

Node *getActiveParent(Network &network, const Node *node) {
  assert(node);
  if (node->type == NodeType::RETICULATION_NODE) {
    return getReticulationActiveParent(network, node);
  }
  for (const auto &link : node->links) {
    if (link.direction == Direction::INCOMING) {
      return getTargetNode(network, &link);
    }
  }
  return nullptr;
}

std::vector<Node *> getAllParents(Network &network, const Node *node) {
  assert(node);
  std::vector<Node *> res;
  if (node->type == NodeType::RETICULATION_NODE) {
    res.emplace_back(getReticulationFirstParent(network, node));
    res.emplace_back(getReticulationSecondParent(network, node));
  } else {
    Node *parent = getActiveParent(network, node);
    if (parent) {
      res.emplace_back(parent);
    }
  }
  return res;
}

std::vector<Node *> getParentPointers(
    AnnotatedNetwork &ann_network,
    const std::vector<ReticulationState> &reticulationChoices,
    const Node *virtual_root) {
  assert(virtual_root);
  setReticulationParents(ann_network.network, reticulationChoices);
  std::vector<Node *> parent(ann_network.network.num_nodes(), nullptr);
  parent[virtual_root->clv_index] =
      ann_network.network.nodes_by_index[virtual_root->clv_index];
  std::queue<Node *> q;

  q.emplace(ann_network.network.nodes_by_index[virtual_root->clv_index]);
  while (!q.empty()) {
    Node *actNode = q.front();
    q.pop();
    std::vector<Node *> neighbors =
        getActiveNeighbors(ann_network.network, actNode);
    for (size_t i = 0; i < neighbors.size(); ++i) {
      Node *neigh = neighbors[i];
      if (!parent[neigh->clv_index]) {  // neigh was not already processed
        q.emplace(neigh);
        parent[neigh->clv_index] = actNode;
      }
    }
  }
  parent[virtual_root->clv_index] = nullptr;
  return parent;
}

std::vector<Node *> getParentPointers(AnnotatedNetwork &ann_network,
                                      const Node *virtual_root) {
  assert(virtual_root);
  std::vector<Node *> parent(ann_network.network.num_nodes(), nullptr);
  parent[virtual_root->clv_index] =
      ann_network.network.nodes_by_index[virtual_root->clv_index];
  std::queue<Node *> q;

  q.emplace(ann_network.network.nodes_by_index[virtual_root->clv_index]);
  while (!q.empty()) {
    Node *actNode = q.front();
    q.pop();
    std::vector<Node *> neighbors =
        getActiveNeighbors(ann_network.network, actNode);
    for (size_t i = 0; i < neighbors.size(); ++i) {
      Node *neigh = neighbors[i];
      if (!parent[neigh->clv_index]) {  // neigh was not already processed
        q.emplace(neigh);
        parent[neigh->clv_index] = actNode;
      }
    }
  }
  parent[virtual_root->clv_index] = nullptr;
  return parent;
}

std::vector<Node *> grab_current_node_parents(Network &network) {
  std::vector<Node *> parent(network.num_nodes(), nullptr);
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    parent[i] = getActiveParent(network, network.nodes_by_index[i]);
  }
  return parent;
}

}  // namespace netrax