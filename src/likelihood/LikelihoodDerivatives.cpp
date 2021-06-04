#include "LikelihoodDerivatives.hpp"
#include "../graph/NodeDisplayedTreeData.hpp"
#include "../helper/Helper.hpp"
#include "mpreal.h"

namespace netrax {

struct TreeDerivatives {
  mpfr::mpreal lh_prime = 0.0;
  mpfr::mpreal lh_prime_prime = 0.0;
};

TreeDerivatives computeTreeDerivatives(double logl, double logl_prime,
                                       double logl_prime_prime) {
  TreeDerivatives res;
  mpfr::mpreal lh = mpfr::exp(logl);
  mpfr::mpreal lh_prime = lh * logl_prime;
  mpfr::mpreal lh_prime_prime = lh_prime * logl_prime + lh * logl_prime_prime;

  res.lh_prime = lh_prime;
  res.lh_prime_prime = lh_prime_prime;
  return res;
}

struct PartitionLhData {
  double logl_prime = 0.0;
  double logl_prime_prime = 0.0;
};

PartitionLhData computePartitionLhData(
    AnnotatedNetwork &ann_network, unsigned int partition_idx,
    const std::vector<SumtableInfo> &sumtables, unsigned int pmatrix_index) {
  PartitionLhData res{0.0, 0.0};
  Node *source = getSource(ann_network.network,
                           ann_network.network.edges_by_index[pmatrix_index]);
  Node *target = getTarget(ann_network.network,
                           ann_network.network.edges_by_index[pmatrix_index]);

  bool single_tree_mode = (sumtables.size() == 1);

  // TODO: Get NetRAX to correctly work with scaled brlens...
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
    throw std::runtime_error(
        "I believe this function currently does not work correctly with scaled "
        "branch lengths");
  }

  double s = 1.0;
  // double s = ann_network.fake_treeinfo->brlen_scalers ?
  // ann_network.fake_treeinfo->brlen_scalers[partition_idx] : 1.;
  double p_brlen =
      s *
      ann_network.fake_treeinfo->branch_lengths[partition_idx][pmatrix_index];

  // mpfr::mpreal logl = 0.0;
  mpfr::mpreal lh_sum = 0.0;
  mpfr::mpreal lh_prime_sum = 0.0;
  mpfr::mpreal lh_prime_prime_sum = 0.0;

  double best_tree_logl_score = -std::numeric_limits<double>::infinity();
  double best_tree_logl_prime_score = -std::numeric_limits<double>::infinity();
  double best_tree_logl_prime_prime_score =
      -std::numeric_limits<double>::infinity();

  double branch_length;
  if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
    branch_length =
        ann_network.fake_treeinfo->branch_lengths[partition_idx][pmatrix_index];
  } else {
    branch_length =
        ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index];
  }

  double **eigenvals = nullptr;
  double *prop_invar = nullptr;
  double *diagptable = nullptr;

  if (ann_network.fake_treeinfo->partitions[partition_idx]) {
    pll_partition_t *partition =
        ann_network.fake_treeinfo->partitions[partition_idx];
    pll_compute_eigenvals_and_prop_invar(
        partition, ann_network.fake_treeinfo->param_indices[partition_idx],
        &eigenvals, &prop_invar);
    diagptable = pll_compute_diagptable(partition->states, partition->rate_cats,
                                        branch_length, prop_invar,
                                        partition->rates, eigenvals);
    free(eigenvals);
  }

  for (size_t i = 0; i < sumtables.size(); ++i) {
    double tree_logl = 0.0;
    double tree_logl_prime = 0.0;
    double tree_logl_prime_prime = 0.0;

    if (ann_network.fake_treeinfo->partitions[partition_idx]) {
      pll_partition_t *partition =
          ann_network.fake_treeinfo->partitions[partition_idx];
      pll_compute_loglikelihood_derivatives(
          partition, source->scaler_index,
          sumtables[i].left_tree->scale_buffer[partition_idx],
          target->scaler_index,
          sumtables[i].right_tree->scale_buffer[partition_idx], p_brlen,
          ann_network.fake_treeinfo->param_indices[partition_idx],
          sumtables[i].sumtable, (single_tree_mode) ? nullptr : &tree_logl,
          &tree_logl_prime, &tree_logl_prime_prime, diagptable, prop_invar);
    }

    /* sum up values from all threads */
    if (ann_network.fake_treeinfo->parallel_reduce_cb) {
      if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        if (!single_tree_mode) {
          ann_network.fake_treeinfo->parallel_reduce_cb(
              ann_network.fake_treeinfo->parallel_context, &tree_logl,
              ann_network.fake_treeinfo->partition_count,
              PLLMOD_COMMON_REDUCE_SUM);
        }
        ann_network.fake_treeinfo->parallel_reduce_cb(
            ann_network.fake_treeinfo->parallel_context, &tree_logl_prime,
            ann_network.fake_treeinfo->partition_count,
            PLLMOD_COMMON_REDUCE_SUM);
        ann_network.fake_treeinfo->parallel_reduce_cb(
            ann_network.fake_treeinfo->parallel_context, &tree_logl_prime_prime,
            ann_network.fake_treeinfo->partition_count,
            PLLMOD_COMMON_REDUCE_SUM);
      } else {
        if (single_tree_mode) {
          double d[2] = {tree_logl_prime, tree_logl_prime_prime};
          ann_network.fake_treeinfo->parallel_reduce_cb(
              ann_network.fake_treeinfo->parallel_context, d, 2,
              PLLMOD_COMMON_REDUCE_SUM);
          tree_logl_prime = d[0];
          tree_logl_prime_prime = d[1];
        } else {
          double d[3] = {tree_logl, tree_logl_prime, tree_logl_prime_prime};
          ann_network.fake_treeinfo->parallel_reduce_cb(
              ann_network.fake_treeinfo->parallel_context, d, 3,
              PLLMOD_COMMON_REDUCE_SUM);
          tree_logl = d[0];
          tree_logl_prime = d[1];
          tree_logl_prime_prime = d[2];
        }
      }
    }

    if (single_tree_mode) {
      pll_aligned_free(diagptable);
      free(prop_invar);
      res.logl_prime = tree_logl_prime;
      res.logl_prime_prime = tree_logl_prime_prime;
      return res;
    }

    if (ann_network.options.likelihood_variant ==
        LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
      TreeDerivatives treeDerivatives = computeTreeDerivatives(
          tree_logl, tree_logl_prime, tree_logl_prime_prime);
      lh_sum += mpfr::exp(tree_logl) * sumtables[i].tree_prob;
      lh_prime_sum += treeDerivatives.lh_prime * sumtables[i].tree_prob;
      lh_prime_prime_sum +=
          treeDerivatives.lh_prime_prime * sumtables[i].tree_prob;
    } else {  // LikelihoodVariant::BEST_DISPLAYED_TREE
      if (tree_logl * sumtables[i].tree_prob > best_tree_logl_score) {
        best_tree_logl_score = tree_logl * sumtables[i].tree_prob;
        best_tree_logl_prime_score = tree_logl_prime;
        best_tree_logl_prime_prime_score = tree_logl_prime_prime;
      }
    }
  }

  pll_aligned_free(diagptable);
  free(prop_invar);

  if (ann_network.options.likelihood_variant ==
      LikelihoodVariant::AVERAGE_DISPLAYED_TREES) {
    res.logl_prime = (lh_prime_sum / lh_sum).toDouble();
    res.logl_prime_prime =
        ((lh_prime_prime_sum * lh_sum - lh_prime_sum * lh_prime_sum) /
         (lh_sum * lh_sum))
            .toDouble();
  } else {  // LikelihoodVariant::BEST_DISPLAYED_TREE
    res.logl_prime = best_tree_logl_prime_score;
    res.logl_prime_prime = best_tree_logl_prime_prime_score;
  }

  // res.lh_prime *= s;
  // res.lh_prime_prime *= s * s;
  return res;
}

LoglDerivatives computeLoglikelihoodDerivatives(
    AnnotatedNetwork &ann_network,
    const std::vector<std::vector<SumtableInfo>> &sumtables,
    unsigned int pmatrix_index) {
  // setup_pmatrices(ann_network, incremental, update_pmatrices);
  // double network_logl = 0.0;
  double network_logl_prime = 0.0;
  double network_logl_prime_prime = 0.0;
  assert(sumtables.size() == ann_network.fake_treeinfo->partition_count);
  std::vector<double> partition_logls_prime(
      ann_network.fake_treeinfo->partition_count, 0.0);
  std::vector<double> partition_logls_prime_prime(
      ann_network.fake_treeinfo->partition_count, 0.0);

  for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count;
       ++p) {  // here we need to go over all partitions, as the derivatives
               // require exact tree loglikelihood
    PartitionLhData pdata =
        computePartitionLhData(ann_network, p, sumtables[p], pmatrix_index);

    // std::cout << " Network partition loglikelihood derivatives for partition
    // " << p << ":\n"; std::cout << " partition_logl: " << pdata.logl << "\n";
    // std::cout << " partition_logl_prime: " << pdata.logl_prime << "\n";
    // std::cout << " partition_logl_prime_prime: " << pdata.logl_prime_prime <<
    // "\n";

    partition_logls_prime[p] = pdata.logl_prime;
    partition_logls_prime_prime[p] = pdata.logl_prime_prime;

    // network_logl += pdata.logl;
    network_logl_prime += pdata.logl_prime;
    network_logl_prime_prime += pdata.logl_prime_prime;
  }

  // std::cout << "Network loglikelihood derivatives:\n";
  // std::cout << "network_logl: " << network_logl << "\n";
  // std::cout << "network_logl_prime: " << network_logl_prime << "\n";
  // std::cout << "network_logl_prime_prime: " << network_logl_prime_prime <<
  // "\n"; std::cout << "network_logl_prime / network_logl_prime_prime: " <<
  // network_logl_prime / network_logl_prime_prime << "\n"; std::cout << "\n";
  return LoglDerivatives{network_logl_prime, network_logl_prime_prime,
                         partition_logls_prime, partition_logls_prime_prime};
}

/*double computeLoglikelihoodFromSumtables(AnnotatedNetwork& ann_network, const
std::vector<std::vector<SumtableInfo> >& sumtables, const
std::vector<std::vector<TreeLoglData> >& oldTrees, unsigned int pmatrix_index,
bool incremental, bool update_pmatrices) { setup_pmatrices(ann_network,
incremental, update_pmatrices); mpfr::mpreal network_logl = 0.0;

assert(sumtables.size() == ann_network.fake_treeinfo->partition_count);

for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
    PartitionLhData pdata = computePartitionLhData(ann_network, p, sumtables[p],
oldTrees, pmatrix_index); network_logl += pdata.logl;
}
return network_logl.toDouble();
}*/

SumtableInfo computeSumtable(
    AnnotatedNetwork &ann_network, size_t partition_idx,
    const ReticulationConfigSet &restrictions, DisplayedTreeData &left_tree,
    size_t left_clv_index, DisplayedTreeData &right_tree,
    size_t right_clv_index, size_t left_tree_idx, size_t right_tree_idx) {
  // skip remote partitions
  if (!ann_network.fake_treeinfo
           ->partitions[partition_idx]) {  // add an empty fake sumtable
    SumtableInfo sumtableInfo(0, 0, &left_tree, &right_tree, left_tree_idx,
                              right_tree_idx);
    sumtableInfo.tree_prob = computeReticulationConfigProb(
        restrictions, ann_network.first_parent_logprobs,
        ann_network.second_parent_logprobs);
    return sumtableInfo;
  }

  pll_partition_t *partition =
      ann_network.fake_treeinfo->partitions[partition_idx];
  size_t sumtableSize = (partition->sites + partition->states) *
                        partition->rate_cats * partition->states_padded;
  SumtableInfo sumtableInfo(sumtableSize, partition->alignment, &left_tree,
                            &right_tree, left_tree_idx, right_tree_idx);

  sumtableInfo.tree_prob = computeReticulationConfigProb(
      restrictions, ann_network.first_parent_logprobs,
      ann_network.second_parent_logprobs);
  sumtableInfo.sumtable = (double *)pll_aligned_alloc(
      sumtableSize * sizeof(double), partition->alignment);
  if (!sumtableInfo.sumtable) {
    throw std::runtime_error("Error in allocating memory for sumtable");
  }
  pll_update_sumtable(partition, left_clv_index,
                      left_tree.clv_vector[partition_idx], right_clv_index,
                      right_tree.clv_vector[partition_idx],
                      left_tree.scale_buffer[partition_idx],
                      right_tree.scale_buffer[partition_idx],
                      ann_network.fake_treeinfo->param_indices[partition_idx],
                      sumtableInfo.sumtable);

  return sumtableInfo;
}

std::vector<std::vector<SumtableInfo>> computePartitionSumtables(
    AnnotatedNetwork &ann_network, unsigned int pmatrix_index) {
  std::vector<std::vector<SumtableInfo>> res(
      ann_network.fake_treeinfo->partition_count);
  Node *source = getSource(ann_network.network,
                           ann_network.network.edges_by_index[pmatrix_index]);
  Node *target = getTarget(ann_network.network,
                           ann_network.network.edges_by_index[pmatrix_index]);

  size_t n_trees_source =
      ann_network.pernode_displayed_tree_data[source->clv_index]
          .num_active_displayed_trees;
  size_t n_trees_target =
      ann_network.pernode_displayed_tree_data[target->clv_index]
          .num_active_displayed_trees;
  std::vector<DisplayedTreeData> &sourceTrees =
      ann_network.pernode_displayed_tree_data[source->clv_index]
          .displayed_trees;
  std::vector<DisplayedTreeData> &targetTrees =
      ann_network.pernode_displayed_tree_data[target->clv_index]
          .displayed_trees;

  for (size_t i = 0; i < n_trees_source; ++i) {
    for (size_t j = 0; j < n_trees_target; ++j) {
      if (!reticulationConfigsCompatible(
              sourceTrees[i].treeLoglData.reticulationChoices,
              targetTrees[j].treeLoglData.reticulationChoices)) {
        continue;
      }

      ReticulationConfigSet restrictions = combineReticulationChoices(
          sourceTrees[i].treeLoglData.reticulationChoices,
          targetTrees[j].treeLoglData.reticulationChoices);
      if (isActiveBranch(ann_network, restrictions, pmatrix_index)) {
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count;
             ++p) {  // here we need all partitions, as we require the sumtable
                     // info metadata
          res[p].emplace_back(std::move(computeSumtable(
              ann_network, p, restrictions, sourceTrees[i], source->clv_index,
              targetTrees[j], target->clv_index, i, j)));
        }
      }
    }
  }

  // TODO: This is just for comining left and right tree. Sometimes, we need a
  // single tree...
  return res;
}

}  // namespace netrax