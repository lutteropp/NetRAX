#include "VirtualRerooting.hpp"
#include "LikelihoodComputation.hpp"
#include "../helper/Helper.hpp"
#include "../DebugPrintFunctions.hpp"

#include "ImprovedLoglikelihood.hpp"

namespace netrax {
struct PathToVirtualRoot {
    ReticulationConfigSet reticulationChoices;
    std::vector<Node*> path;
    std::vector<std::vector<Node*>> children;

    PathToVirtualRoot(size_t max_reticulations) : reticulationChoices(max_reticulations) {};
};

std::vector<Node*> getPathToVirtualRoot(Node* from, Node* virtual_root, const std::vector<Node*> parent) {
    assert(from);
    assert(virtual_root);
    std::vector<Node*> res;
    Node* act_node = from;
    while (act_node != virtual_root){
        res.emplace_back(act_node);
        act_node = parent[act_node->clv_index];
    }
    res.emplace_back(virtual_root);
    return res;
}

void printPathToVirtualRoot(const PathToVirtualRoot& pathToVirtualRoot) {
    std::cout << "Path has reticulation choices:\n";
    printReticulationChoices(pathToVirtualRoot.reticulationChoices);
    for (size_t i = 0; i < pathToVirtualRoot.path.size(); ++i) {
        std::cout << "Node " << pathToVirtualRoot.path[i]->clv_index << " has children: ";
        for (size_t j = 0; j < pathToVirtualRoot.children[i].size(); ++j) {
            std::cout << pathToVirtualRoot.children[i][j]->clv_index;
            if (j + 1 < pathToVirtualRoot.children[i].size()) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    }
}

std::vector<PathToVirtualRoot> getPathsToVirtualRoot(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back) {
    std::vector<PathToVirtualRoot> res;
    // naive version here: Go over all displayed trees, compute pathToVirtualRoot for each of them, and then later on kick out duplicate paths...

    NodeDisplayedTreeData& oldDisplayedTrees = ann_network.pernode_displayed_tree_data[old_virtual_root->clv_index];
    for (size_t i = 0; i < oldDisplayedTrees.num_active_displayed_trees; ++i) {
        PathToVirtualRoot ptvr(ann_network.options.max_reticulations);
        for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
            setReticulationParents(ann_network.network, oldDisplayedTrees.displayed_trees[i].treeLoglData.reticulationChoices.configs[0]);
        }
        std::vector<Node*> parent = getParentPointers(ann_network, new_virtual_root);
        std::vector<Node*> path = getPathToVirtualRoot(old_virtual_root, new_virtual_root, parent);
        
        std::vector<ReticulationState> dont_care_vector(ann_network.options.max_reticulations, ReticulationState::DONT_CARE);
        ReticulationConfigSet restrictionsSet(ann_network.options.max_reticulations);
        restrictionsSet.configs.emplace_back(dont_care_vector);

        for (size_t j = 0; j < path.size() - 1; ++j) {
            restrictionsSet = combineReticulationChoices(restrictionsSet, getRestrictionsToTakeNeighbor(ann_network, path[j], path[j+1]));
        }
        assert(restrictionsSet.configs.size() == 1);
        ptvr.reticulationChoices = restrictionsSet;
        ptvr.path = path;

        assert(ptvr.path[0] == old_virtual_root);
        for (size_t j = 0; j < path.size() - 1; ++j) {
            if (path[j] == new_virtual_root_back) {
                ptvr.children.emplace_back(getCurrentChildren(ann_network, path[j], new_virtual_root, restrictionsSet)); // Not sure if this special case is needed
            } else {
                ptvr.children.emplace_back(getCurrentChildren(ann_network, path[j], parent[path[j]->clv_index], restrictionsSet));
            }
        }
        // Special case at new virtual root, there we don't want to include new_virtual_root_back...
        assert(path[path.size() - 1] == new_virtual_root);
        ptvr.children.emplace_back(getCurrentChildren(ann_network, new_virtual_root, new_virtual_root_back, restrictionsSet));

        res.emplace_back(ptvr);
    }

    // Kick out the duplicate paths
    bool foundDuplicate = true;
    while (foundDuplicate) {
        foundDuplicate = false;
        for (size_t i = 0; i < res.size() - 1; ++i) {
            for (size_t j = i + 1; j < res.size(); ++j) {
                if (res[i].path == res[j].path) {
                    foundDuplicate = true;
                    std::swap(res[j], res[res.size() - 1]);
                    res.pop_back();
                    break;
                }
            }
            if (foundDuplicate) {
                break;
            }
        }
    }

    return res;
}

struct NodeSaveInformation {
    std::vector<std::unordered_set<size_t> > pathNodesToRestore;
    std::unordered_set<size_t> nodesInDanger;
};

NodeSaveInformation computeNodeSaveInformation(const std::vector<PathToVirtualRoot>& paths) {
    NodeSaveInformation nodeSaveInfo;
    std::vector<std::unordered_set<size_t> >& pathNodesToRestore = nodeSaveInfo.pathNodesToRestore;
    std::unordered_set<size_t>& nodesInDanger = nodeSaveInfo.nodesInDanger;
    
    pathNodesToRestore.resize(paths.size());
    for (size_t p = 1; p < paths.size(); ++p) {
        std::unordered_set<size_t> nodesInPath;
        for (size_t i = 0; i < paths[p].path.size(); ++i) {
            nodesInPath.emplace(paths[p].path[i]->clv_index);
            for (size_t j = 0; j < paths[p].children[i].size(); ++j) {
                pathNodesToRestore[p].emplace(paths[p].children[i][j]->clv_index);
            }
        }
        for (size_t nodeInPath : nodesInPath) {
            pathNodesToRestore[p].erase(nodeInPath);
        }

        // check if the node occurs at an earlier path. If so, it really needs to be restored, as that earlier path would have overwritten it.
        std::unordered_set<size_t> deleteAgain;

        for (size_t maybeSaveMe : pathNodesToRestore[p]) {
            bool saveMe = false;
            for (size_t q = 0; q < p; ++q) {
                for (size_t i = 0; i < paths[q].path.size(); ++i) {
                    if (paths[q].path[i]->clv_index == maybeSaveMe) {
                        saveMe = true;
                        break;
                    }
                }
                if (saveMe) {
                    break;
                }
            }
            if (!saveMe) {
                deleteAgain.emplace(maybeSaveMe);
            }
        }

        for (size_t deleteMeAgain : deleteAgain) {
            pathNodesToRestore[p].erase(deleteMeAgain);
        }
    }

    for (size_t p = 0; p < paths.size(); ++p) {
        for (size_t nodeToSave : pathNodesToRestore[p]) {
            nodesInDanger.emplace(nodeToSave);
        }
    }

    return nodeSaveInfo;
}

void updateCLVsVirtualRerootTrees(AnnotatedNetwork& ann_network, Node* old_virtual_root, Node* new_virtual_root, Node* new_virtual_root_back) {
    assert(old_virtual_root);
    assert(new_virtual_root);

    // 1.) for all paths from retNode to new_virtual_root:
    //     1.1) Collect the reticulation nodes encountered on the path, build exta restrictions storing the reticulation configurations used
    //     1.2) update CLVs on that path, using extra restrictions and append mode
    std::vector<PathToVirtualRoot> paths = getPathsToVirtualRoot(ann_network, old_virtual_root, new_virtual_root, new_virtual_root_back);

    // figure out which nodes to save from old CLVs, and which paths need them
    NodeSaveInformation nodeSaveInfo = computeNodeSaveInformation(paths);
    std::vector<NodeDisplayedTreeData> bufferedNodeInformations(ann_network.network.num_nodes());
    for (size_t partition_idx = 0; partition_idx < ann_network.fake_treeinfo->partition_count; ++partition_idx) {
        for (size_t nodeInDanger : nodeSaveInfo.nodesInDanger) {
            bufferedNodeInformations[nodeInDanger] = ann_network.pernode_displayed_tree_data[nodeInDanger];
        }
    }
    
    for (size_t p = 0; p < paths.size(); ++p) {
        /*std::cout << "PROCESSING PATH " << p << " ON PARTITION " << partition_idx << "\n";
        printPathToVirtualRoot(paths[p]);
        std::cout << "The path has the following restrictions: \n";
        printReticulationChoices(paths[p].reticulationChoices);*/

        // Restore required old NodeInformations for the path
        for (size_t nodeIndexToRestore : nodeSaveInfo.pathNodesToRestore[p]) {
            /*if (ann_network.network.num_reticulations() == 1) {
                std::cout << "restoring node info at " << nodeIndexToRestore << "\n";
            }*/
            ann_network.pernode_displayed_tree_data[nodeIndexToRestore] = bufferedNodeInformations[nodeIndexToRestore];
        }

        for (size_t i = 0; i < paths[p].path.size(); ++i) {
            bool appendMode = ((p > 0) && (paths[p].path[i] == new_virtual_root));
            assert((paths[p].path[i] != new_virtual_root) || ((paths[p].path[i] == new_virtual_root) && (i == paths[p].path.size() - 1)));
            processNodeImproved(ann_network, 0, paths[p].path[i], paths[p].children[i], paths[p].reticulationChoices, appendMode);
        }
    }
}

void updateTreeData(AnnotatedNetwork& ann_network, const std::vector<DisplayedTreeData>& oldTrees, TreeLoglData& treeData) {
    const TreeLoglData& oldTree = getMatchingTreeData(oldTrees, treeData.reticulationChoices);
    assert(oldTree.tree_logl_valid);
    treeData.tree_partition_logl = oldTree.tree_partition_logl;
    assert(oldTree.tree_logprob_valid);
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        assert(oldTree.tree_partition_logl[p] <= 0.0);
    }
    treeData.tree_logprob = computeReticulationConfigLogProb(treeData.reticulationChoices, ann_network.reticulation_probs);
    treeData.tree_logl_valid = true;
    treeData.tree_logprob_valid = true;
}

void recomputeTreeData(AnnotatedNetwork& ann_network, size_t pmatrix_index, DisplayedTreeData& sourceTree, DisplayedTreeData& targetTree, TreeLoglData& combinedTreeData) {
    Node* source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        combinedTreeData.tree_partition_logl[p] = 0.0;
        // skip remote partitions
        if (!ann_network.fake_treeinfo->partitions[p]) {
            continue;
        }
        pll_partition_t* partition = ann_network.fake_treeinfo->partitions[p];
        std::vector<double> persite_logl(ann_network.fake_treeinfo->partitions[p]->sites);
        assert(sourceTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[p], sourceTree.clv_vector[p]));
        assert(targetTree.isTip || !single_clv_is_all_zeros(ann_network.partition_clv_ranges[p], targetTree.clv_vector[p]));
        combinedTreeData.tree_partition_logl[p] = pll_compute_edge_loglikelihood(partition, source->clv_index, sourceTree.clv_vector[p], sourceTree.scale_buffer[p], 
                                                    target->clv_index, targetTree.clv_vector[p], targetTree.scale_buffer[p], 
                                                    pmatrix_index, ann_network.fake_treeinfo->param_indices[p], persite_logl.data());
        assert(combinedTreeData.tree_partition_logl[p] != -std::numeric_limits<double>::infinity());
        assert(combinedTreeData.tree_partition_logl[p] < 0.0);
    }

    /* sum up likelihood from all threads */
    if (ann_network.fake_treeinfo->parallel_reduce_cb)
    {
        ann_network.fake_treeinfo->parallel_reduce_cb(ann_network.fake_treeinfo->parallel_context,
                                    combinedTreeData.tree_partition_logl.data(),
                                    ann_network.fake_treeinfo->partition_count,
                                    PLLMOD_COMMON_REDUCE_SUM);

        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            if (combinedTreeData.tree_partition_logl[p] >= 0.0) {
                std::cout << "thread " << ParallelContext::local_proc_id() << " combinedTreeData.tree_partition_logl[" << p << "]: " << combinedTreeData.tree_partition_logl[p] << "\n";
            }
            if (combinedTreeData.tree_partition_logl[p] == 0.0) {
                throw std::runtime_error("bad partition logl");
            }
            assert(combinedTreeData.tree_partition_logl[p] < 0.0);
        }
    }
    combinedTreeData.tree_logprob = computeReticulationConfigLogProb(combinedTreeData.reticulationChoices, ann_network.reticulation_probs);
}

double computeLoglikelihoodBrlenOpt(AnnotatedNetwork &ann_network, const std::vector<DisplayedTreeData>& oldTrees, unsigned int pmatrix_index, int incremental, int update_pmatrices) {
    if (ann_network.cached_logl_valid) {
        return ann_network.cached_logl;
    }
    Node* source = getSource(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    Node* target = getTarget(ann_network.network, ann_network.network.edges_by_index[pmatrix_index]);
    NodeDisplayedTreeData& sourceData =  ann_network.pernode_displayed_tree_data[source->clv_index];
    NodeDisplayedTreeData& targetData =  ann_network.pernode_displayed_tree_data[target->clv_index];
    std::vector<DisplayedTreeData>& sourceTrees = sourceData.displayed_trees;
    std::vector<DisplayedTreeData>& targetTrees = targetData.displayed_trees;
    size_t n_trees_source = sourceData.num_active_displayed_trees;
    size_t n_trees_target = targetData.num_active_displayed_trees;
    std::vector<bool> source_tree_seen(n_trees_source, false);
    std::vector<bool> target_tree_seen(n_trees_target, false);

    assert(reuseOldDisplayedTreesCheck(ann_network, incremental)); // TODO: Doesn't this need the virtual_root pointer, too?
    if (update_pmatrices) {
        pllmod_treeinfo_update_prob_matrices(ann_network.fake_treeinfo, !incremental);
    }
    //Instead of going over the-source-trees-only for final loglh evaluation, we need to go over all pairs of trees, one in source node and one in target node.
    std::vector<TreeLoglData> combinedTrees;

    for (size_t i = 0; i < n_trees_source; ++i) {
        for (size_t j = 0; j < n_trees_target; ++j) {
            if (!reticulationConfigsCompatible(sourceTrees[i].treeLoglData.reticulationChoices, targetTrees[j].treeLoglData.reticulationChoices)) {
                continue;
            }
            TreeLoglData combinedTreeData(ann_network.fake_treeinfo->partition_count, ann_network.options.max_reticulations);
            combinedTreeData.reticulationChoices = combineReticulationChoices(sourceTrees[i].treeLoglData.reticulationChoices, targetTrees[j].treeLoglData.reticulationChoices);
            if (isActiveBranch(ann_network, combinedTreeData.reticulationChoices, pmatrix_index)) {
                recomputeTreeData(ann_network, pmatrix_index, sourceTrees[i], targetTrees[j], combinedTreeData);
            } else {
                updateTreeData(ann_network, oldTrees, combinedTreeData);
            }
            combinedTreeData.tree_logl_valid = true;
            combinedTreeData.tree_logprob_valid = true;
            combinedTrees.emplace_back(combinedTreeData);

            source_tree_seen[i] = true;
            target_tree_seen[j] = true;
        }
    }

    for (size_t i = 0; i < n_trees_source; ++i) {
        if (!source_tree_seen[i]) {
            updateTreeData(ann_network, oldTrees, sourceTrees[i].treeLoglData);
            combinedTrees.emplace_back(sourceTrees[i].treeLoglData);
        }
    }
    for (size_t j = 0; j < n_trees_target; ++j) {
        if (!target_tree_seen[j]) {
            updateTreeData(ann_network, oldTrees, targetTrees[j].treeLoglData);
            combinedTrees.emplace_back(targetTrees[j].treeLoglData);
        }
    }

    for (size_t c = 0; c < combinedTrees.size(); ++c) {
        assert(combinedTrees[c].tree_logl_valid);
    }
    double network_logl = 0;
    for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        network_logl += evaluateTreesPartition(ann_network, p, combinedTrees);
    }
    ann_network.cached_logl = network_logl;
    ann_network.cached_logl_valid = true;

    /*if (ParallelContext::local_proc_id() == 0) {
        std::cout << "combined trees:\n";
        for (size_t i = 0; i < combinedTrees.size(); ++i) {
            printReticulationChoices(combinedTrees[i].reticulationChoices);
            for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
                std::cout << "  partition_loglh[" << p << "]: " << combinedTrees[i].tree_partition_logl[p] << "\n";
            }
        }
    }*/

    return network_logl;
}

}