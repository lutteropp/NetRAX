#include "Helper.hpp"

#include "../NetraxOptions.hpp"
#include "../graph/NodeDisplayedTreeData.hpp"

namespace netrax {

void invalidateSingleClv(AnnotatedNetwork &ann_network,
                         unsigned int clv_index) {
  assert(clv_index >= ann_network.network.num_tips());
  pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
  for (size_t p = 0; p < treeinfo->partition_count; ++p) {
    // skip remote partitions
    if (!treeinfo->partitions[p]) {
      continue;
    }
    treeinfo->clv_valid[p][clv_index] = 0;
  }
  for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[clv_index]
                             .num_active_displayed_trees;
       ++i) {
    ann_network.pernode_displayed_tree_data[clv_index]
        .displayed_trees[i]
        .clv_valid = false;
    ann_network.pernode_displayed_tree_data[clv_index]
        .displayed_trees[i]
        .treeLoglData.tree_logl_valid = false;
  }

  // TODO: This is commented out because it broke things. Find out why it breaks
  // things.
  ann_network.pernode_displayed_tree_data[clv_index]
      .num_active_displayed_trees = 0;
  if (ann_network.options.save_memory) {
    ann_network.pernode_displayed_tree_data[clv_index].displayed_trees.clear();
  }
  ann_network.pseudo_clv_valid[clv_index] = false;
  ann_network.cached_logl_valid = false;
}

void validateSingleClv(AnnotatedNetwork &ann_network, unsigned int clv_index) {
  pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
  for (size_t p = 0; p < treeinfo->partition_count; ++p) {
    // skip remote partitions
    if (!treeinfo->partitions[p]) {
      continue;
    }
    treeinfo->clv_valid[p][clv_index] = 1;
  }
}

void invalidateHigherClvs(AnnotatedNetwork &ann_network,
                          pllmod_treeinfo_t *treeinfo, const Node *node,
                          bool invalidate_myself, std::vector<bool> &visited) {
  Network &network = ann_network.network;
  if (!node) {
    return;
  }
  if (node->clv_index < ann_network.network.num_tips()) {
    invalidate_myself = false;
  }
  if (!visited.empty() &&
      visited[node->clv_index]) {  // clv at node is already invalidated
    return;
  }
  if (invalidate_myself) {
    invalidateSingleClv(ann_network, node->clv_index);
    if (!visited.empty()) {
      visited[node->clv_index] = true;
    }
  }
  if (node->clv_index == network.root->clv_index) {
    return;
  }
  if (node->type == NodeType::RETICULATION_NODE) {
    invalidateHigherClvs(ann_network, treeinfo,
                         getReticulationFirstParent(network, node), true,
                         visited);
    invalidateHigherClvs(ann_network, treeinfo,
                         getReticulationSecondParent(network, node), true,
                         visited);
  } else {
    invalidateHigherClvs(ann_network, treeinfo, getActiveParent(network, node),
                         true, visited);
  }
  ann_network.cached_logl_valid = false;
}

void invalidateHigherPseudoClvs(AnnotatedNetwork &ann_network,
                                pllmod_treeinfo_t *treeinfo, const Node *node,
                                bool invalidate_myself,
                                std::vector<bool> &visited) {
  Network &network = ann_network.network;
  if (!node) {
    return;
  }
  if (node->clv_index < ann_network.network.num_tips()) {
    invalidate_myself = false;
  }
  if (!visited.empty() &&
      visited[node->clv_index]) {  // clv at node is already invalidated
    return;
  }
  if (invalidate_myself) {
    ann_network.pseudo_clv_valid[node->clv_index] = false;
    if (!visited.empty()) {
      visited[node->clv_index] = true;
    }
  }
  if (node->clv_index == network.root->clv_index) {
    return;
  }
  if (node->type == NodeType::RETICULATION_NODE) {
    invalidateHigherClvs(ann_network, treeinfo,
                         getReticulationFirstParent(network, node), true,
                         visited);
    invalidateHigherClvs(ann_network, treeinfo,
                         getReticulationSecondParent(network, node), true,
                         visited);
  } else {
    invalidateHigherClvs(ann_network, treeinfo, getActiveParent(network, node),
                         true, visited);
  }
  ann_network.cached_logl_valid = false;
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, const Node *node,
                          bool invalidate_myself, std::vector<bool> &visited) {
  pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
  invalidateHigherClvs(ann_network, treeinfo, node, invalidate_myself, visited);
}

void invalidateHigherPseudoCLVs(AnnotatedNetwork &ann_network, const Node *node,
                                bool invalidate_myself,
                                std::vector<bool> &visited) {
  pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
  invalidateHigherPseudoClvs(ann_network, treeinfo, node, invalidate_myself,
                             visited);
}

void invalidateHigherCLVs(AnnotatedNetwork &ann_network, const Node *node,
                          bool invalidate_myself) {
  pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
  std::vector<bool> noVisited;
  invalidateHigherClvs(ann_network, treeinfo, node, invalidate_myself,
                       noVisited);
}

void invalidateHigherPseudoCLVs(AnnotatedNetwork &ann_network, Node *node,
                                bool invalidate_myself) {
  pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
  std::vector<bool> noVisited;
  invalidateHigherPseudoClvs(ann_network, treeinfo, node, invalidate_myself,
                             noVisited);
}

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network, size_t pmatrix_index,
                            std::vector<bool> &visited) {
  pllmod_treeinfo_t *treeinfo = ann_network.fake_treeinfo;
  for (size_t p = 0; p < treeinfo->partition_count; ++p) {
    // skip remote partitions
    if (!treeinfo->partitions[p]) {
      continue;
    }
    treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
  }
  invalidateHigherCLVs(
      ann_network,
      getSource(ann_network.network,
                ann_network.network.edges_by_index[pmatrix_index]),
      true, visited);
}

void invalidatePmatrixIndex(AnnotatedNetwork &ann_network,
                            size_t pmatrix_index) {
  std::vector<bool> noVisited;
  invalidatePmatrixIndex(ann_network, pmatrix_index, noVisited);
}

void invalidPmatrixIndexOnly(AnnotatedNetwork &ann_network,
                             size_t pmatrix_index) {
  for (size_t partition_idx = 0;
       partition_idx < ann_network.fake_treeinfo->partition_count;
       ++partition_idx) {
    // skip remote partitions
    if (!ann_network.fake_treeinfo->partitions[partition_idx]) {
      continue;
    }
    ann_network.fake_treeinfo->pmatrix_valid[partition_idx][pmatrix_index] = 0;
  }
  ann_network.cached_logl_valid = false;
}

bool allClvsValid(AnnotatedNetwork &ann_network, size_t clv_index) {
  for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
    if (ann_network.fake_treeinfo->partitions[p]) {
      if (!ann_network.fake_treeinfo->clv_valid[p][clv_index]) {
        return false;
      }
    }
  }
  if (ann_network.pernode_displayed_tree_data[clv_index].num_active_displayed_trees == 0) {
    return false;
  }
  for (size_t i = 0; i < ann_network.pernode_displayed_tree_data[clv_index]
                             .num_active_displayed_trees;
       ++i) {
    DisplayedTreeData &dtd =
        ann_network.pernode_displayed_tree_data[clv_index].displayed_trees[i];
    if (!dtd.treeLoglData.tree_logprob_valid) {
      dtd.treeLoglData.tree_logprob =
          computeReticulationConfigLogProb(dtd.treeLoglData.reticulationChoices,
                                           ann_network.first_parent_logprobs,
                                           ann_network.second_parent_logprobs);
      if (dtd.treeLoglData.reticulationChoices.configs[0][0] !=
          ReticulationState::DONT_CARE) {
        assert(dtd.treeLoglData.tree_logprob != 0.0);
      }
      dtd.treeLoglData.tree_logprob_valid = true;
    }
    if (!dtd.clv_valid &&
        dtd.treeLoglData.tree_logprob <
            ann_network.options.min_interesting_tree_logprob) {
      return false;
    }
  }
  return true;
}

void invalidate_pmatrices(AnnotatedNetwork &ann_network,
                          std::vector<size_t> &affectedPmatrixIndices) {
  pllmod_treeinfo_t *fake_treeinfo = ann_network.fake_treeinfo;
  for (size_t pmatrix_index : affectedPmatrixIndices) {
    assert(ann_network.network.edges_by_index[pmatrix_index]);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
      // skip reote partitions
      if (!fake_treeinfo->partitions[p]) {
        continue;
      }
      fake_treeinfo->pmatrix_valid[p][pmatrix_index] = 0;
    }
  }
  pllmod_treeinfo_update_prob_matrices(fake_treeinfo, 0);
}

void invalidateAllCLVs(AnnotatedNetwork &ann_network) {
  for (size_t i = ann_network.network.num_tips();
       i < ann_network.network.num_nodes(); ++i) {
    invalidateSingleClv(ann_network, i);
  }
}

bool hasReticulationChoice(DisplayedTreeData &dtd, size_t reticulation_idx) {
  for (size_t i = 0; i < dtd.treeLoglData.reticulationChoices.configs.size();
       ++i) {
    if (dtd.treeLoglData.reticulationChoices.configs[i][reticulation_idx] !=
        ReticulationState::DONT_CARE) {
      return true;
    }
  }
  return false;
}

void invalidateTreeLogprobs(AnnotatedNetwork &ann_network,
                            size_t reticulation_idx) {
  for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
    for (size_t j = 0;
         j <
         ann_network.pernode_displayed_tree_data[i].num_active_displayed_trees;
         ++j) {
      DisplayedTreeData &dtd =
          ann_network.pernode_displayed_tree_data[i].displayed_trees[j];
      if (hasReticulationChoice(dtd, reticulation_idx)) {
        dtd.treeLoglData.tree_logprob = computeReticulationConfigLogProb(
            dtd.treeLoglData.reticulationChoices,
            ann_network.first_parent_logprobs,
            ann_network.second_parent_logprobs);
        if (dtd.treeLoglData.reticulationChoices.configs[0][0] !=
            ReticulationState::DONT_CARE) {
          assert(dtd.treeLoglData.tree_logprob != 0.0);
        }
        dtd.treeLoglData.tree_logprob_valid = true;
      }
    }
  }
}

void invalidateTreeLogprobs(AnnotatedNetwork &ann_network) {
  for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
    invalidateTreeLogprobs(ann_network, i);
  }
}

}  // namespace netrax