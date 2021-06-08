#include "Helper.hpp"

#include "../DebugPrintFunctions.hpp"
#include "../NetraxOptions.hpp"
#include "../graph/NodeDisplayedTreeData.hpp"
#include "NetworkFunctions.hpp"

namespace netrax {

void setReticulationState(Network &network, size_t reticulation_idx,
                          ReticulationState state) {
  assert(reticulation_idx < network.reticulation_nodes.size());
  if (state == ReticulationState::DONT_CARE) {
    return;
  } else if (state == ReticulationState::TAKE_FIRST_PARENT) {
    network.reticulation_nodes[reticulation_idx]
        ->getReticulationData()
        ->setActiveParentToggle(0);
  } else {  // TAKE_SECOND_PARENT
    network.reticulation_nodes[reticulation_idx]
        ->getReticulationData()
        ->setActiveParentToggle(1);
  }
}

ReticulationConfigSet getRestrictionsToDismissNeighbor(
    AnnotatedNetwork &ann_network, const Node *node, const Node *neighbor) {
  ReticulationConfigSet res(ann_network.options.max_reticulations);
  std::vector<ReticulationState> restrictions(
      ann_network.options.max_reticulations, ReticulationState::DONT_CARE);
  assert(node);
  assert(neighbor);
  bool foundRestriction = false;
  if (node->getType() == NodeType::RETICULATION_NODE) {
    if (neighbor == getReticulationFirstParent(ann_network.network, node)) {
      restrictions[node->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_SECOND_PARENT;
      foundRestriction = true;
    } else if (neighbor ==
               getReticulationSecondParent(ann_network.network, node)) {
      restrictions[node->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_FIRST_PARENT;
      foundRestriction = true;
    }
  }
  if (neighbor->getType() == NodeType::RETICULATION_NODE) {
    if (node == getReticulationFirstParent(ann_network.network, neighbor)) {
      restrictions[neighbor->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_SECOND_PARENT;
      foundRestriction = true;
    } else if (node ==
               getReticulationSecondParent(ann_network.network, neighbor)) {
      restrictions[neighbor->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_FIRST_PARENT;
      foundRestriction = true;
    }
  }
  if (foundRestriction) {
    res.configs.emplace_back(restrictions);
  }
  return res;
}

ReticulationConfigSet getRestrictionsToTakeNeighbor(
    AnnotatedNetwork &ann_network, const Node *node, const Node *neighbor) {
  ReticulationConfigSet res(ann_network.options.max_reticulations);
  std::vector<ReticulationState> restrictions(
      ann_network.options.max_reticulations, ReticulationState::DONT_CARE);
  assert(node);
  assert(neighbor);
  if (node->getType() == NodeType::RETICULATION_NODE) {
    if (neighbor == getReticulationFirstParent(ann_network.network, node)) {
      restrictions[node->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_FIRST_PARENT;
    } else if (neighbor ==
               getReticulationSecondParent(ann_network.network, node)) {
      restrictions[node->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_SECOND_PARENT;
    }
  }
  if (neighbor->getType() == NodeType::RETICULATION_NODE) {
    if (node == getReticulationFirstParent(ann_network.network, neighbor)) {
      restrictions[neighbor->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_FIRST_PARENT;
    } else if (node ==
               getReticulationSecondParent(ann_network.network, neighbor)) {
      restrictions[neighbor->getReticulationData()->reticulation_index] =
          ReticulationState::TAKE_SECOND_PARENT;
    }
  }
  res.configs.emplace_back(restrictions);
  return res;
}

ReticulationConfigSet getReticulationChoicesThisOnly(
    AnnotatedNetwork &ann_network,
    const ReticulationConfigSet &this_tree_config,
    const ReticulationConfigSet &other_child_dead_settings, const Node *parent,
    const Node *this_child, const Node *other_child) {
  // covers both dead children and reticulation children
  ReticulationConfigSet res(ann_network.options.max_reticulations);

  ReticulationConfigSet this_reachable_from_parent_restrictionSet =
      getRestrictionsToTakeNeighbor(ann_network, parent, this_child);
  ReticulationConfigSet restrictedConfig = combineReticulationChoices(
      this_tree_config, this_reachable_from_parent_restrictionSet);

  if (restrictedConfig.configs
          .empty()) {  // easy case, this_tree isn't reachable anyway
    return res;
  }

  // Find all configurations where we can take restricted_config, but we cannot
  // take any of the trees from displayed_trees_other

  // easy case: parent not being an active parent of other_child
  ReticulationConfigSet other_not_reachable_from_parent_restrictionSet =
      getRestrictionsToDismissNeighbor(ann_network, parent, other_child);
  ReticulationConfigSet combinedConfig = combineReticulationChoices(
      restrictedConfig, other_not_reachable_from_parent_restrictionSet);
  for (size_t i = 0; i < combinedConfig.configs.size(); ++i) {
    res.configs.emplace_back(combinedConfig.configs[i]);
  }

  ReticulationConfigSet other_reachable_from_parent_restrictionSet =
      getRestrictionsToTakeNeighbor(ann_network, parent, other_child);
  restrictedConfig = combineReticulationChoices(
      restrictedConfig, other_reachable_from_parent_restrictionSet);
  // now in restrictedConfig, we have the case where parent has two active
  // children, plus we have this_tree on the left. We need to check if there are
  // configurations where other_child is a dead node.
  ReticulationConfigSet combinedConfig2 =
      combineReticulationChoices(restrictedConfig, other_child_dead_settings);

  for (size_t i = 0; i < combinedConfig2.configs.size(); ++i) {
    res.configs.emplace_back(combinedConfig2.configs[i]);
  }

  simplifyReticulationChoices(res);
  return res;
}

ReticulationConfigSet deadNodeSettings(
    AnnotatedNetwork &ann_network, const NodeDisplayedTreeData &displayed_trees,
    const Node *parent, const Node *child) {
  // Return all configurations in which the node which the displayed trees
  // belong to would have no displayed tree, and thus be a dead node
  ReticulationConfigSet res(ann_network.options.max_reticulations);

  ReticulationConfigSet childNotTakenRestriction =
      getRestrictionsToDismissNeighbor(ann_network, parent, child);
  for (size_t i = 0; i < childNotTakenRestriction.configs.size(); ++i) {
    res.configs.emplace_back(childNotTakenRestriction.configs[i]);
  }
  ReticulationConfigSet childTakenRestriction =
      getRestrictionsToTakeNeighbor(ann_network, parent, child);

  if (ann_network.network.num_reticulations() > sizeof(size_t) * 8) {
    throw std::runtime_error(
        "This implementation only works for <= sizeof(size_t)*8 reticulations");
  }
  size_t max_n_trees = (1 << ann_network.network.num_reticulations());
  for (size_t tree_idx = 0; tree_idx < max_n_trees; ++tree_idx) {
    ReticulationConfigSet reticulationChoices =
        getTreeConfig(ann_network, tree_idx);
    if (!reticulationConfigsCompatible(reticulationChoices,
                                       childTakenRestriction)) {
      continue;
    }
    bool foundTree = false;
    for (size_t i = 0; i < displayed_trees.num_active_displayed_trees; ++i) {
      if (reticulationConfigsCompatible(
              reticulationChoices, displayed_trees.displayed_trees[i]
                                       .treeLoglData.reticulationChoices)) {
        foundTree = true;
        break;
      }
    }
    if (!foundTree) {
      res.configs.emplace_back(reticulationChoices.configs[0]);
    }
  }

  simplifyReticulationChoices(res);
  return res;
}

ReticulationConfigSet getTreeConfig(AnnotatedNetwork &ann_network,
                                    size_t tree_idx) {
  std::vector<ReticulationState> reticulationChoicesVector(
      ann_network.options.max_reticulations);
  ReticulationConfigSet reticulationChoices(
      ann_network.options.max_reticulations);
  reticulationChoices.configs.emplace_back(reticulationChoicesVector);
  for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
    if (tree_idx & (1 << i)) {
      reticulationChoices.configs[0][i] = ReticulationState::TAKE_SECOND_PARENT;
    } else {
      reticulationChoices.configs[0][i] = ReticulationState::TAKE_FIRST_PARENT;
    }
  }
  return reticulationChoices;
}

DisplayedTreeData &findMatchingDisplayedTree(
    AnnotatedNetwork &ann_network,
    const ReticulationConfigSet &reticulationChoices,
    NodeDisplayedTreeData &data) {
  DisplayedTreeData *tree = nullptr;

  size_t n_good = 0;
  for (size_t i = 0; i < data.num_active_displayed_trees; ++i) {
    if (reticulationConfigsCompatible(
            reticulationChoices,
            data.displayed_trees[i].treeLoglData.reticulationChoices)) {
      n_good++;
      tree = &data.displayed_trees[i];
    }
  }
  if (n_good == 1) {
    return *tree;
  } else if (n_good > 1) {
    std::cout << exportDebugInfo(ann_network) << "\n";
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
      std::cout << "displayed trees stored at node " << i << ":\n";
      for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[i]
                                 .num_active_displayed_trees;
           ++j) {
        printReticulationChoices(ann_network.pernode_displayed_tree_data[i]
                                     .displayed_trees[j]
                                     .treeLoglData.reticulationChoices);
      }
    }
    std::cout << "Reticulation first parents:\n";
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
      std::cout << "reticulation node "
                << ann_network.network.reticulation_nodes[i]->clv_index
                << " has first parent "
                << getReticulationFirstParent(
                       ann_network.network,
                       ann_network.network.reticulation_nodes[i])
                       ->clv_index
                << "\n";
    }
    throw std::runtime_error("Found multiple suitable trees");
  } else {  // n_good == 0
    throw std::runtime_error("Found no suitable displayed tree");
  }
}

Node *findFirstNodeWithTwoActiveChildren(
    AnnotatedNetwork &ann_network,
    const ReticulationConfigSet &reticulationChoices, const Node *oldRoot) {
  // TODO: Make this work with direction-agnistic stuff (virtual rerooting)
  // throw std::runtime_error("TODO: Make this work with direction-agnistic
  // stuff (virtual rerooting)");

  // all these reticulation choices led to the same tree, thus it is safe to
  // simply use the first one for detecting which nodes to skip...
  for (size_t i = 0; i < reticulationChoices.configs[0].size();
       ++i) {  // apply the reticulation choices
    if (reticulationChoices.configs[0][i] != ReticulationState::DONT_CARE) {
      setReticulationState(ann_network.network, i,
                           reticulationChoices.configs[0][i]);
    }
  }

  Node *displayed_tree_root = nullptr;
  collect_dead_nodes(ann_network.network, oldRoot->clv_index,
                     &displayed_tree_root);
  return displayed_tree_root;
}

DisplayedTreeData &getMatchingDisplayedTreeAtNode(
    AnnotatedNetwork &ann_network, unsigned int node_clv_index,
    const ReticulationConfigSet &queryChoices) {
  for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[node_clv_index]
                             .num_active_displayed_trees;
       ++i) {
    if (reticulationConfigsCompatible(
            queryChoices,
            ann_network.pernode_displayed_tree_data[node_clv_index]
                .displayed_trees[i]
                .treeLoglData.reticulationChoices)) {
      return ann_network.pernode_displayed_tree_data[node_clv_index]
          .displayed_trees[i];
    }
  }
  throw std::runtime_error("No compatible displayed tree data found");
}

const TreeLoglData &getMatchingTreeData(
    const std::vector<DisplayedTreeData> &trees,
    const ReticulationConfigSet &queryChoices) {
  for (size_t i = 0; i < trees.size(); ++i) {
    if (reticulationConfigsCompatible(
            queryChoices, trees[i].treeLoglData.reticulationChoices)) {
      return trees[i].treeLoglData;
    }
  }
  std::cout << "query was:\n";
  printReticulationChoices(queryChoices);
  throw std::runtime_error("No compatible old tree data found");
}

ReticulationConfigSet getRestrictionsActiveBranch(AnnotatedNetwork &ann_network,
                                                  size_t pmatrix_index) {
  ReticulationConfigSet res;
  for (size_t tree_idx = 0;
       tree_idx < (1 << ann_network.network.num_reticulations()); ++tree_idx) {
    ReticulationConfigSet treeChoices = getTreeConfig(ann_network, tree_idx);
    if (isActiveBranch(ann_network, treeChoices, pmatrix_index)) {
      res.configs.emplace_back(treeChoices.configs[0]);
    }
  }
  simplifyReticulationChoices(res);
  return res;
}

ReticulationConfigSet getRestrictionsActiveAliveBranch(
    AnnotatedNetwork &ann_network, size_t pmatrix_index) {
  ReticulationConfigSet res;
  for (size_t tree_idx = 0;
       tree_idx < (1 << ann_network.network.num_reticulations()); ++tree_idx) {
    ReticulationConfigSet treeChoices = getTreeConfig(ann_network, tree_idx);
    if (isActiveAliveBranch(ann_network, treeChoices, pmatrix_index)) {
      res.configs.emplace_back(treeChoices.configs[0]);
    }
  }
  simplifyReticulationChoices(res);
  return res;
}

}  // namespace netrax