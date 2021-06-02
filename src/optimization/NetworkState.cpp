#include <algorithm>

#include "../NetraxOptions.hpp"
#include "../graph/DisplayedTreeData.hpp"
#include "../helper/Helper.hpp"
#include "../helper/NetworkFunctions.hpp"
#include "../likelihood/LikelihoodComputation.hpp"
#include "NetworkState.hpp"

namespace netrax {

bool assert_tip_links(const Network &network) {
  for (size_t i = 0; i < network.num_tips(); ++i) {
    assert(network.nodes_by_index[i]->clv_index == i);
    assert(!network.nodes_by_index[i]->label.empty());
    assert(network.nodes_by_index[i]->links.size() == 1);
  }
  return true;
}

bool consecutive_indices(const Network &network) {
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    assert(network.nodes_by_index[i]);
  }
  for (size_t i = 0; i < network.num_branches(); ++i) {
    assert(network.edges_by_index[i]);
  }
  return true;
}

bool assert_links_in_range(const Network &network) {
  for (size_t i = 0; i < network.num_nodes(); ++i) {
    for (size_t j = 0; j < network.nodes_by_index[i]->links.size(); ++j) {
      assert(network.nodes_by_index[i]->links[j].edge_pmatrix_index <
             network.num_branches());
    }
  }
  for (size_t i = 0; i < network.num_branches(); ++i) {
    assert(network.edges_by_index[i]->link1->edge_pmatrix_index == i);
    assert(network.edges_by_index[i]->link2->edge_pmatrix_index == i);
  }
  return true;
}

bool assert_branch_lengths(AnnotatedNetwork &ann_network) {
  if (ann_network.fake_treeinfo->brlen_linkage ==
      PLLMOD_COMMON_BRLEN_UNLINKED) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
        assert(ann_network.fake_treeinfo->branch_lengths[p][i] >=
               ann_network.options.brlen_min);
        assert(ann_network.fake_treeinfo->branch_lengths[p][i] <=
               ann_network.options.brlen_max);
      }
    }
  } else {
    for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
      assert(ann_network.fake_treeinfo->linked_branch_lengths[i] >=
             ann_network.options.brlen_min);
      assert(ann_network.fake_treeinfo->linked_branch_lengths[i] <=
             ann_network.options.brlen_max);
    }
  }
  return true;
}

bool neighborsSame(Network &n1, Network &n2) {
  bool same = true;
  for (size_t i = 0; i < n1.num_nodes(); ++i) {
    std::vector<Node *> n1_neighbors = getNeighbors(n1, n1.nodes_by_index[i]);
    std::vector<size_t> n1_neigh_indices(n1_neighbors.size());
    for (size_t j = 0; j < n1_neighbors.size(); ++j) {
      n1_neigh_indices[j] = n1_neighbors[j]->clv_index;
    }
    std::sort(n1_neigh_indices.begin(), n1_neigh_indices.end());

    std::vector<Node *> n2_neighbors = getNeighbors(n2, n2.nodes_by_index[i]);
    std::vector<size_t> n2_neigh_indices(n2_neighbors.size());
    for (size_t j = 0; j < n2_neighbors.size(); ++j) {
      n2_neigh_indices[j] = n2_neighbors[j]->clv_index;
    }
    std::sort(n2_neigh_indices.begin(), n2_neigh_indices.end());

    same &= (n1_neigh_indices == n2_neigh_indices);
  }
  return same;
}

void extract_network_state(AnnotatedNetwork &ann_network,
                           NetworkState &state_to_reuse, bool update_model) {
  assert(assert_tip_links(ann_network.network));
  assert(assert_links_in_range(ann_network.network));
  state_to_reuse.brlen_linkage = ann_network.options.brlen_linkage;

  // branch lengths stuff
  state_to_reuse.linked_brlens.resize(
      ann_network.fake_treeinfo->tree->edge_count);
  for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
    assert(ann_network.fake_treeinfo->linked_branch_lengths[i] >=
               ann_network.options.brlen_min &&
           ann_network.fake_treeinfo->linked_branch_lengths[i] <=
               ann_network.options.brlen_max);
    state_to_reuse.linked_brlens[i] =
        ann_network.fake_treeinfo->linked_branch_lengths[i];
  }
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
    for (size_t p = 0; p < state_to_reuse.partition_brlens.size(); ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      for (size_t pmatrix_index = 0;
           pmatrix_index < ann_network.network.num_branches();
           ++pmatrix_index) {
        assert(ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] >=
                   ann_network.options.brlen_min &&
               ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] <=
                   ann_network.options.brlen_max);
        state_to_reuse.partition_brlens[p][pmatrix_index] =
            ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
      }
    }
  }
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      state_to_reuse.partition_brlen_scalers[p] =
          ann_network.fake_treeinfo->brlen_scalers[p];
    }
  }

  if (update_model) {
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      state_to_reuse.alphas[p] = ann_network.fake_treeinfo->alphas[p];
    }

    for (size_t i = 0; i < state_to_reuse.partition_models.size(); ++i) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[i]) {
        continue;
      }
      assign(state_to_reuse.partition_models[i],
             ann_network.fake_treeinfo->partitions[i]);
    }
  }
  state_to_reuse.reticulation_probs = ann_network.reticulation_probs;
  state_to_reuse.n_trees = (1 << ann_network.network.num_reticulations());
  state_to_reuse.n_branches = ann_network.network.num_branches();

  state_to_reuse.cached_logl = ann_network.cached_logl;
  state_to_reuse.cached_logl_valid = ann_network.cached_logl_valid;
}

NetworkState extract_network_state(AnnotatedNetwork &ann_network) {
  NetworkState state(true);

  // branch lengths allocation stuff
  state.linked_brlens.resize(ann_network.fake_treeinfo->tree->edge_count);
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
    state.partition_brlens.resize(ann_network.fake_treeinfo->partition_count);
    for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      state.partition_brlens[p].resize(ann_network.network.edges.size());
    }
  }
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
    state.partition_brlen_scalers.resize(
        ann_network.fake_treeinfo->partition_count);
  }

  state.alphas.resize(ann_network.fake_treeinfo->partition_count);
  state.partition_models.resize(ann_network.fake_treeinfo->partition_count);

  extract_network_state(ann_network, state, true);

  return state;
}

bool assert_rates(AnnotatedNetwork &ann_network) {
  for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
    // skip remote partitions
    if (!ann_network.fake_treeinfo->partitions[p]) {
      continue;
    }
    std::vector<double> recomputed_rates(
        ann_network.fake_treeinfo->partitions[p]->rate_cats);
    pll_compute_gamma_cats(ann_network.fake_treeinfo->alphas[p],
                           ann_network.fake_treeinfo->partitions[p]->rate_cats,
                           recomputed_rates.data(),
                           ann_network.fake_treeinfo->gamma_mode[p]);
    for (size_t k = 0; k < recomputed_rates.size(); ++k) {
      assert(ann_network.fake_treeinfo->partitions[p]->rates[k] ==
             recomputed_rates[k]);
    }
  }
  return true;
}

void apply_network_state(AnnotatedNetwork &ann_network,
                         const NetworkState &state, bool update_model) {
  assert(ann_network.fake_treeinfo);
  // assert(computeLoglikelihood(ann_network) ==
  // computeLoglikelihood(ann_network, 0, 1)); ann_network.options.brlen_linkage
  // = state.brlen_linkage;

  // branch lengths stuff
  assert(ann_network.fake_treeinfo->tree->edge_count ==
         state.linked_brlens.size());
  for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
    if (ann_network.fake_treeinfo->linked_branch_lengths[i] !=
        state.linked_brlens[i]) {
      assert(state.linked_brlens[i] >= ann_network.options.brlen_min &&
             state.linked_brlens[i] <= ann_network.options.brlen_max);
      ann_network.fake_treeinfo->linked_branch_lengths[i] =
          state.linked_brlens[i];
      invalidatePmatrixIndex(ann_network, i);
    }
  }
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
    for (size_t p = 0; p < state.partition_brlens.size(); ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      for (size_t pmatrix_index = 0;
           pmatrix_index < ann_network.network.num_branches();
           ++pmatrix_index) {
        if (ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] !=
            state.partition_brlens[p][pmatrix_index]) {
          assert(state.partition_brlens[p][pmatrix_index] >=
                     ann_network.options.brlen_min &&
                 state.partition_brlens[p][pmatrix_index] <=
                     ann_network.options.brlen_max);
          ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] =
              state.partition_brlens[p][pmatrix_index];
          invalidatePmatrixIndex(ann_network, pmatrix_index);
        }
      }
    }
  }
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
    for (size_t p = 0; p < state.partition_brlen_scalers.size(); ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      ann_network.fake_treeinfo->brlen_scalers[p] =
          state.partition_brlen_scalers[p];
    }
  }
  for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
    if (ann_network.reticulation_probs[i] != state.reticulation_probs[i]) {
      assert(state.reticulation_probs[i] >= ann_network.options.brprob_min &&
             state.reticulation_probs[i] <= ann_network.options.brprob_max);
      ann_network.reticulation_probs[i] = state.reticulation_probs[i];
      ann_network.cached_logl_valid = false;
    }
  }

  if (update_model) {
    for (size_t p = 0; p < state.alphas.size(); ++p) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[p]) {
        continue;
      }
      ann_network.fake_treeinfo->alphas[p] = state.alphas[p];
    }
    for (size_t i = 0; i < state.partition_models.size(); ++i) {
      // skip remote partitions
      if (!ann_network.fake_treeinfo->partitions[i]) {
        continue;
      }
      assign(ann_network.fake_treeinfo->partitions[i],
             state.partition_models[i]);
    }

    pllmod_treeinfo_update_prob_matrices(
        ann_network.fake_treeinfo,
        1);  // this (full pmatrix recomputation) is needed if the model
             // parameters changed
    invalidateAllCLVs(ann_network);
  }

  ann_network.cached_logl_valid = false;
  assert(assert_branch_lengths(ann_network));
  assert(assert_rates(ann_network));
}

bool reticulation_probs_equal(const NetworkState &old_state,
                              const NetworkState &act_state) {
  assert(old_state.reticulation_probs.size() ==
         act_state.reticulation_probs.size());
  bool all_fine = true;
  for (size_t j = 0; j < old_state.reticulation_probs.size(); ++j) {
    if (fabs(act_state.reticulation_probs[j] -
             old_state.reticulation_probs[j]) >= 1E-5) {
      std::cout << "wanted prob:\n";
      std::cout << "idx " << j << ": " << old_state.reticulation_probs[j]
                << "\n";
      std::cout << "\n";
      std::cout << "observed prob:\n";
      std::cout << "idx " << j << ": " << act_state.reticulation_probs[j]
                << "\n";
      std::cout << "\n";

      std::cout << "reticulation probs not equal\n";
      all_fine = false;
      break;
    }
  }
  return all_fine;
}

bool brlen_scalers_equal(const NetworkState &old_state,
                         const NetworkState &act_state) {
  assert(old_state.partition_brlen_scalers.size() ==
         act_state.partition_brlen_scalers.size());
  for (size_t i = 0; i < act_state.partition_brlen_scalers.size(); ++i) {
    if (fabs(act_state.partition_brlen_scalers[i] -
             old_state.partition_brlen_scalers[i]) >= 1E-5) {
      std::cout << "brlen scalers not equal\n";
      return false;
    }
  }
  return true;
}

bool partition_brlens_equal(const NetworkState &old_state,
                            const NetworkState &act_state) {
  assert(old_state.partition_brlens.size() ==
         act_state.partition_brlens.size());
  assert(old_state.n_branches == act_state.n_branches);
  bool all_fine = true;
  for (size_t i = 0; i < old_state.partition_brlens.size(); ++i) {
    assert(act_state.partition_brlens[i].size() ==
           old_state.partition_brlens[i].size());
    for (size_t j = 0; j < act_state.partition_brlens[i].size(); ++j) {
      if (fabs(act_state.partition_brlens[i][j] -
               old_state.partition_brlens[i][j]) >= 1E-5) {
        std::cout << "wanted brlens:\n";
        for (size_t k = 0; k < old_state.partition_brlens[i].size(); ++k) {
          std::cout << "idx " << k << ": " << old_state.partition_brlens[i][k]
                    << "\n";
        }
        std::cout << "\n";
        std::cout << "observed brlens:\n";
        for (size_t k = 0; k < act_state.partition_brlens[i].size(); ++k) {
          std::cout << "idx " << k << ": " << act_state.partition_brlens[i][k]
                    << "\n";
        }
        std::cout << "\n";
        std::cout << "brlens not equal\n";
        all_fine = false;
      }
    }
  }
  return all_fine;
}

bool model_equal(const NetworkState &old_state, const NetworkState &act_state) {
  if (old_state.partition_models.size() != act_state.partition_models.size()) {
    return false;
  }
  for (size_t p = 0; p < old_state.partition_models.size(); ++p) {
    if (old_state.partition_models[p].to_string(true) !=
        act_state.partition_models[p].to_string(true)) {
      std::cout << "model not equal\n";
      return false;
    }
  }
  return true;
}

bool alphas_equal(const NetworkState &old_state,
                  const NetworkState &act_state) {
  if (old_state.alphas.size() != act_state.alphas.size()) {
    return false;
  }
  for (size_t i = 0; i < old_state.alphas.size(); ++i) {
    if (old_state.alphas[i] != act_state.alphas[i]) {
      std::cout << "alphas not equal\n";
      return false;
    }
  }
  return true;
}

bool network_states_equal(const NetworkState &old_state,
                          const NetworkState &act_state) {
  return model_equal(old_state, act_state) &&
         reticulation_probs_equal(old_state, act_state) &&
         partition_brlens_equal(old_state, act_state) &&
         alphas_equal(old_state, act_state) &&
         brlen_scalers_equal(old_state, act_state);
}

AnnotatedNetwork build_annotated_network_from_state(
    NetworkState &state, const NetraxOptions &options) {
  throw std::runtime_error("Not implemented yet");
}

}  // namespace netrax
