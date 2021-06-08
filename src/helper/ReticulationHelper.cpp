#include "../NetraxOptions.hpp"
#include "Helper.hpp"

namespace netrax {

Node *getReticulationChild(Network &network, const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return getTargetNode(network, node->getReticulationData()->getLinkToChild());
}

Node *getReticulationFirstParent(Network &network, const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return getTargetNode(network,
                       node->getReticulationData()->link_to_first_parent);
}

Node *getReticulationSecondParent(Network &network, const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return getTargetNode(network,
                       node->getReticulationData()->link_to_second_parent);
}

Node *getReticulationActiveParent(Network &network, const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return getTargetNode(network,
                       node->getReticulationData()->getLinkToActiveParent());
}

Node *getReticulationNonActiveParent(Network &network, const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return getTargetNode(network,
                       node->getReticulationData()->getLinkToNonActiveParent());
}

double getReticulationFirstParentProb(AnnotatedNetwork &ann_network,
                                      const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return ann_network
      .reticulation_probs[node->getReticulationData()->reticulation_index];
}

double getReticulationSecondParentProb(AnnotatedNetwork &ann_network,
                                       const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return 1.0 - ann_network.reticulation_probs[node->getReticulationData()
                                                  ->reticulation_index];
}

double getReticulationActiveProb(AnnotatedNetwork &ann_network,
                                 const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  size_t first_parent_pmatrix_index =
      getReticulationFirstParentPmatrixIndex(node);
  size_t active_pmatrix_index =
      node->getReticulationData()->getLinkToActiveParent()->edge_pmatrix_index;

  if (first_parent_pmatrix_index == active_pmatrix_index) {
    return getReticulationFirstParentProb(ann_network, node);
  } else {
    return getReticulationSecondParentProb(ann_network, node);
  }
}

size_t getReticulationFirstParentPmatrixIndex(const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  assert(node->getReticulationData());
  assert(node->getReticulationData()->getLinkToFirstParent());
  return node->getReticulationData()
      ->getLinkToFirstParent()
      ->edge_pmatrix_index;
}

size_t getReticulationSecondParentPmatrixIndex(const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return node->getReticulationData()
      ->getLinkToSecondParent()
      ->edge_pmatrix_index;
}

size_t getReticulationActiveParentPmatrixIndex(const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return node->getReticulationData()
      ->getLinkToActiveParent()
      ->edge_pmatrix_index;
}

size_t getReticulationChildPmatrixIndex(const Node *node) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  return node->getReticulationData()->getLinkToChild()->edge_pmatrix_index;
}

Node *getReticulationOtherParent(Network &network, const Node *node,
                                 const Node *parent) {
  assert(node);
  assert(node->type == NodeType::RETICULATION_NODE);
  if (getReticulationFirstParent(network, node) == parent) {
    return getReticulationSecondParent(network, node);
  } else {
    assert(getReticulationSecondParent(network, node) == parent);
    return getReticulationFirstParent(network, node);
  }
}

bool assertReticulationProbs(AnnotatedNetwork &ann_network) {
  bool unlinkedMode =
      (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED);
  size_t n_partitions = 1;
  if (unlinkedMode) {
    n_partitions = ann_network.fake_treeinfo->partition_count;
  }
  for (size_t p = 0; p < n_partitions; ++p) {
    for (size_t i = 0; i < ann_network.network.reticulation_nodes.size(); ++i) {
      double actProb = getReticulationActiveProb(
          ann_network, ann_network.network.reticulation_nodes[i]);
      assert(actProb >= 0 && actProb <= 1);
    }
  }
  return true;
}

void setReticulationParents(Network &network, size_t treeIdx) {
  for (size_t i = 0; i < network.num_reticulations(); ++i) {
    // check if i-th bit is set in treeIdx
    bool activeParentIdx = treeIdx & (1 << i);
    network.reticulation_nodes[i]->getReticulationData()->setActiveParentToggle(
        activeParentIdx);
  }
}

void setReticulationParents(
    Network &network,
    const std::vector<ReticulationState> &reticulationChoices) {
  for (size_t i = 0; i < network.num_reticulations(); ++i) {
    if (reticulationChoices[i] == ReticulationState::TAKE_FIRST_PARENT) {
      network.reticulation_nodes[i]
          ->getReticulationData()
          ->setActiveParentToggle(0);
    } else if (reticulationChoices[i] ==
               ReticulationState::TAKE_SECOND_PARENT) {
      network.reticulation_nodes[i]
          ->getReticulationData()
          ->setActiveParentToggle(1);
    }
  }
}

}  // namespace netrax