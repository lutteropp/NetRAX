/*
 * DebugPrintFunctions.cpp
 *
 *  Created on: Jun 4, 2020
 *      Author: sarah
 */

#include "DebugPrintFunctions.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include "graph/AnnotatedNetwork.hpp"
#include "graph/Direction.hpp"
#include "graph/Edge.hpp"
#include "graph/Link.hpp"
#include "graph/Network.hpp"
#include "helper/Helper.hpp"
#include "helper/NetworkFunctions.hpp"
#include "graph/Node.hpp"
#include "graph/NodeType.hpp"
#include "io/RootedNetworkParser.hpp"
#include "io/NetworkIO.hpp"
#include "NetraxOptions.hpp"

#include "graph/NodeDisplayedTreeData.hpp"

namespace netrax {

void printClv(const pllmod_treeinfo_t &treeinfo, size_t clv_index, double* clv, size_t partition_index) {
    size_t sites = treeinfo.partitions[partition_index]->sites;
    size_t rate_cats = treeinfo.partitions[partition_index]->rate_cats;
    size_t states = treeinfo.partitions[partition_index]->states;
    size_t states_padded = treeinfo.partitions[partition_index]->states_padded;
    std::cout << "Clv for clv_index " << clv_index << ": \n";
    for (unsigned int n = 0; n < sites; ++n) {
        for (unsigned int i = 0; i < rate_cats; ++i) {
            for (unsigned int j = 0; j < states; ++j) {
                std::cout
                        << clv[j + i * states_padded] << "\n";
            }
        }
    }
}

void print_clv_vector(pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx, size_t partition_idx,
        size_t clv_index) {
    pll_partition_t *partition = fake_treeinfo.partitions[partition_idx];
    unsigned int states_padded = partition->states_padded;
    unsigned int sites = partition->sites;
    unsigned int rate_cats = partition->rate_cats;
    unsigned int clv_len = states_padded * sites * rate_cats;

    double *clv = partition->clv[clv_index];
    std::cout << "clv vector for tree_idx " << tree_idx << " and clv_index " << clv_index << ":\n";
    for (size_t i = 0; i < clv_len; ++i) {
        std::cout << clv[i] << ",";
    }
    std::cout << "\n";
}

void printOperationArray(const std::vector<pll_operation_t> &ops) {
    for (size_t i = 0; i < ops.size(); ++i) {
        size_t c1 = ops[i].child1_clv_index;
        size_t c2 = ops[i].child2_clv_index;
        size_t p = ops[i].parent_clv_index;
        std::cout << p << " -> " << c1 << ", " << c2 << "\n";
    }
}

void printReticulationParents(Network &network) {
    std::cout << "reticulation parents:\n";
    for (size_t i = 0; i < network.num_reticulations(); ++i) {
        std::cout << "  reticulation " << network.reticulation_nodes[i]->clv_index << " has parent "
                << getActiveParent(network, network.reticulation_nodes[i])->clv_index << "\n";
    }
}

void print_brlens(AnnotatedNetwork &ann_network) {
    std::cout << "brlens:\n";
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
                std::cout << "brlens[" << p << "][" << i << "]: " << ann_network.fake_treeinfo->branch_lengths[p][i] << "\n";
            }
        }
    } else {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            std::cout << "brlens[" << i << "]: " << ann_network.fake_treeinfo->linked_branch_lengths[i] << "\n";
        }
    }
    std::cout << "\n";
}

void printClvValid(AnnotatedNetwork &ann_network) {
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        std::cout << "clv_valid[" << ann_network.network.nodes[i].clv_index << "] = "
                << (int) ann_network.fake_treeinfo->clv_valid[0][ann_network.network.nodes[i].clv_index]
                << "\n";
    }
    std::cout << "\n";
}

void printReticulationFirstParents(AnnotatedNetwork &ann_network) {
    std::cout << "reticulation first parents:\n";
    for (Node *node : ann_network.network.reticulation_nodes) {
        std::cout << "  ret node " << node->clv_index << " has first parent "
                << getReticulationFirstParent(ann_network.network, node)->clv_index << "\n";
    }
}

std::string exportDebugInfoRootedNetwork(const RootedNetwork &rnetwork) {
    std::stringstream ss;
    ss << "graph\n[\tdirected\t1\n";
    for (size_t i = 0; i < rnetwork.nodes.size(); ++i) {
        ss << "\tnode\n\t[\n\t\tid\t" << i << "\n";
        std::string nodeLabel =
                (rnetwork.nodes[i]->label.empty()) ? std::to_string(i) : rnetwork.nodes[i]->label;
        ss << "\t\tlabel\t\"" << nodeLabel << "\"\n";

        ss << "\t\tgraphics\n\t\t[\n";
        ss << "\t\t\tfill\t\"";
        if (rnetwork.nodes[i]->isReticulation) {
            ss << "#00CCFF";
        } else { // normal node
            ss << "#FFCC00";
        }
        ss << "\"\n\t\t]\n";
        ss << "\t]\n";
    }
    for (size_t i = 0; i < rnetwork.nodes.size(); ++i) {
        for (RootedNetworkNode *child : rnetwork.nodes[i]->children) {
            ss << "\tedge\n\t[\n\t\tsource\t";
            unsigned int parentId = i;
            unsigned int childId = std::distance(rnetwork.nodes.begin(),
                    std::find_if(rnetwork.nodes.begin(), rnetwork.nodes.end(),
                            [&](const std::unique_ptr<RootedNetworkNode> &x) {
                                return x.get() == child;
                            }));
            ss << parentId << "\n";
            ss << "\t\ttarget\t" << childId << "\n";
            ss << "\t]\n";
        }
    }
    ss << "]\n";
    return ss.str();
}

std::string buildNodeGraphics(const Node *node) {
    std::stringstream ss;
    ss << "\t\tgraphics\n\t\t[\n";
    ss << "\t\t\tfill\t\"";
    if (node->type == NodeType::RETICULATION_NODE) {
        ss << "#00CCFF";
    } else { // normal node
        ss << "#FFCC00";
    }
    ss << "\"\n\t\t]\n";
    return ss.str();
}

std::string exportDebugInfoExtraNodeNumber(Network &network,
        const std::vector<unsigned int> &extra_node_number, bool with_label) {
    std::stringstream ss;
    ss << "graph\n[\tdirected\t1\n";
    std::vector<Node*> parent = grab_current_node_parents(network);
    for (size_t i = 0; i < network.nodes.size(); ++i) {
        if (network.nodes_by_index[i] == nullptr) {
            continue;
        }
        ss << "\tnode\n\t[\n\t\tid\t" << i << "\n";
        std::string nodeLabel = std::to_string(i);
        if (with_label && !network.nodes_by_index[i]->label.empty()) {
            nodeLabel += ": " + network.nodes_by_index[i]->label;
        }
        if (!extra_node_number.empty()) {
            nodeLabel += "|" + std::to_string(extra_node_number[i]);
        }

        if (network.nodes_by_index[i]->getType() == NodeType::RETICULATION_NODE) {
            size_t firstParentIdx = network.nodes_by_index[i]->getReticulationData()->getLinkToFirstParent()->outer->node_clv_index;
            nodeLabel += "(r=" + std::to_string((network.nodes_by_index[i]->getReticulationData())->getReticulationIndex()) + ", p0=" + std::to_string(firstParentIdx) + ")";
        }

        ss << "\t\tlabel\t\"" << nodeLabel << "\"\n";
        ss << buildNodeGraphics(network.nodes_by_index[i]);
        ss << "\t]\n";
    }
    for (size_t i = 0; i < network.edges.size(); ++i) {
        if (network.edges_by_index[i] == nullptr) {
            continue;
        }
        ss << "\tedge\n\t[\n\t\tsource\t";
        unsigned int parentId = network.edges_by_index[i]->link1->node_clv_index;
        unsigned int childId = network.edges_by_index[i]->link2->node_clv_index;
        if (network.edges_by_index[i]->link1->direction == Direction::INCOMING) {
            std::swap(parentId, childId);
        }
        assert(parentId != childId);
        if (parent[parentId] == network.nodes_by_index[childId]) {
            std::swap(parentId, childId);
        }
        ss << parentId << "\n";
        ss << "\t\ttarget\t" << childId << "\n";
        ss << "\t\tlabel\t";
        std::string edgeLabel = std::to_string(i);
        ss << "\"" << edgeLabel << "\"\n";
        ss << "\t]\n";
    }
    ss << "]\n";
    return ss.str();
}


std::string exportDebugInfo(AnnotatedNetwork &ann_network, bool with_label) {
    //updateNetwork(ann_network);
    return exportDebugInfoExtraNodeNumber(ann_network.network, std::vector<unsigned int>(), with_label);
}

std::string exportDebugInfoNetwork(Network &network, bool with_labels) {
    return exportDebugInfoExtraNodeNumber(network, std::vector<unsigned int>(), with_labels);
}

void print_partition(AnnotatedNetwork& ann_network, pll_partition_t* partition){
    assert(partition);
    std::cout << "\n printing partition...\n";

    size_t imax, jmax;

    std::cout << "clv:\n";
    std::cout << partition->clv << "\n";
    for (size_t clv_index = 0; clv_index < ann_network.network.num_nodes(); ++clv_index) {
        pll_show_clv(partition, clv_index, ann_network.network.nodes_by_index[clv_index]->scaler_index, 10);
    }

    std::cout << "pmatrix:\n";
    for (size_t pmatrix_index = 0; pmatrix_index < ann_network.network.num_branches(); ++pmatrix_index) {
        pll_show_pmatrix(partition, pmatrix_index, 10);
    }
    std::cout << partition->pmatrix << "\n";

    // 2D arrays
    std::cout << "eigenvals:\n";
    imax = partition->rate_matrices;
    jmax = partition->states_padded;
    for (size_t i = 0; i < imax; ++i) {
        for (size_t j = 0; j < jmax; ++j)
        std::cout << partition->eigenvals[i][j] << "\n";
    }
    std::cout << "eigenvecs:\n";
    imax = partition->rate_matrices;
    jmax = partition->states * partition->states_padded;
    for (size_t i = 0; i < imax; ++i) {
        for (size_t j = 0; j < jmax; ++j)
        std::cout << partition->eigenvecs[i][j] << "\n";
    }
    std::cout << "frequencies:\n";
    imax = partition->rate_matrices;
    jmax = partition->states_padded;
    for (size_t i = 0; i < imax; ++i) {
        for (size_t j = 0; j < jmax; ++j)
        std::cout << partition->frequencies[i][j] << "\n";
    }
    std::cout << "inv_eigenvecs:\n";
    imax = partition->rate_matrices;
    jmax = partition->states_padded;
    for (size_t i = 0; i < imax; ++i) {
        for (size_t j = 0; j < jmax; ++j)
        std::cout << partition->inv_eigenvecs[i][j] << "\n";
    }
    std::cout << "scale_buffer:\n";
    imax = partition->scale_buffers;
    unsigned int sites_alloc = (unsigned int) partition->asc_additional_sites + partition->sites;
    size_t scaler_size = (partition->attributes & PLL_ATTRIB_RATE_SCALERS) ?
                                                               sites_alloc * partition->rate_cats : sites_alloc;
    jmax = scaler_size;
    for (size_t i = 0; i < imax; ++i) {
        for (size_t j = 0; j < jmax; ++j)
        std::cout << partition->scale_buffer[i][j] << "\n";
    }
    std::cout << "subst_params:\n";
    imax = partition->rate_matrices;
    jmax = (partition->states * partition->states-partition->states) / 2;
    for (size_t i = 0; i < imax; ++i) {
        for (size_t j = 0; j < jmax; ++j)
        std::cout << partition->subst_params[i][j] << "\n";
    }
    std::cout << "tipchars:\n";
    imax = partition->tips;
    jmax = sites_alloc;
    for (size_t i = 0; i < imax; ++i) {
        for (size_t j = 0; j < jmax; ++j)
        std::cout << partition->tipchars[i][j] << "\n";
    }

    // 1D arrays
    std::cout << "eigen_decomp_valid:\n";
    for (size_t i = 0; i < partition->rate_matrices; ++i) {
        std::cout << partition->eigen_decomp_valid[i] << "\n";
    }
    std::cout << "pattern weights:\n";
    for (size_t i = 0; i < sites_alloc; ++i) {
        std::cout << partition->pattern_weights[i] << "\n";
    }
    std::cout << "prop invar:\n";
    for (size_t i = 0; i < partition->rate_matrices; ++i) {
        std::cout << partition->prop_invar[i] << "\n";
    }
    std::cout << "rate weights:\n";
    for (size_t i = 0; i < partition->rate_cats; ++i) {
        std::cout << partition->rate_weights[i] << "\n";
    }
    std::cout << "rates:\n";
    for (size_t i = 0; i < partition->rate_cats; ++i) {
        std::cout << partition->rates[i] << "\n";
    }
    std::cout << "tipmap:\n";
    for (size_t i = 0; i < PLL_ASCII_SIZE; ++i) {
        std::cout << partition->tipmap[i] << "\n";
    }
    std::cout << "ttlookup:\n";
    for (size_t i = 0; i < 1024 * partition->rate_cats; ++i) {
        std::cout << partition->ttlookup[i] << "\n";
    }

    // single numbers
    std::cout << "alignment: " << partition->alignment << "\n";
    std::cout << "asc_additional_sites: " << partition->asc_additional_sites << "\n";
    std::cout << "asc_bias_alloc: " << partition->asc_bias_alloc << "\n";
    std::cout << "attributes: " << partition->attributes << "\n";
    std::cout << "clv_buffers: " << partition->clv_buffers << "\n";
    std::cout << "maxstates: " << partition->maxstates << "\n";
    std::cout << "nodes: " << partition->nodes << "\n";
    std::cout << "pattern_weight_sum: " << partition->pattern_weight_sum << "\n";
    std::cout << "prob_matrices: " << partition->prob_matrices << "\n";
    std::cout << "rate_cats: " << partition->rate_cats << "\n";
    std::cout << "rate_matrices: " << partition->rate_matrices << "\n";
    std::cout << "scale_buffers: " << partition->scale_buffers << "\n";
    std::cout << "sites: " << partition->sites << "\n";
    std::cout << "states: " << partition->states << "\n";
    std::cout << "states_padded: " << partition->states_padded << "\n";
    std::cout << "tips: " << partition->tips << "\n";
}

void print_treeinfo(AnnotatedNetwork& ann_network) {
    pllmod_treeinfo_t* treeinfo = ann_network.fake_treeinfo;
    // things that stayed the same
    std::cout << "active_partition: " << treeinfo->active_partition << "\n";
    std::cout << "brlen_linkage: " << treeinfo->brlen_linkage << "\n";
    std::cout << "brlen_scalers: " << treeinfo->brlen_scalers << "\n";
    std::cout << "constraint: " << treeinfo->constraint << "\n";
    std::cout << "counter: " << treeinfo->counter << "\n";
    std::cout << "default_likelihood_computation_params: " << treeinfo->default_likelihood_computation_params << "\n";
    std::cout << "default_likelihood_target_function: " << treeinfo->default_likelihood_target_function << "\n";
    std::cout << "init_partition_count: " << treeinfo->init_partition_count << "\n";
    std::cout << "likelihood_target_function: " << treeinfo->likelihood_target_function << "\n";
    std::cout << "matrix_indices: " << treeinfo->matrix_indices << "\n";
    std::cout << "operations: " << treeinfo->operations << "\n";
    std::cout << "parallel_context: " << treeinfo->parallel_context << "\n";
    std::cout << "parallel_reduce_cb: " << treeinfo->parallel_reduce_cb << "\n";
    std::cout << "partition_count: " << treeinfo->partition_count << "\n";
    std::cout << "root: " << treeinfo->root << "\n";
    std::cout << "subnode_count: " << treeinfo->subnode_count << "\n";
    std::cout << "subnodes: " << treeinfo->subnodes << "\n";
    std::cout << "tip_count: " << treeinfo->tip_count << "\n";
    std::cout << "travbuffer: " << treeinfo->travbuffer << "\n";

    // arrays
    std::cout << "alphas:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->alphas[i] << "\n";
    }
    std::cout << "branch_lengths:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->branch_lengths[i] << "\n";
    }
    std::cout << "deriv_precomp:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->deriv_precomp[i] << "\n";
    }
    std::cout << "gamma_mode:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->gamma_mode[i] << "\n";
    }
    std::cout << "init_partition_idx:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->init_partition_idx[i] << "\n";
    }
    std::cout << "init_partitions:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->init_partitions[i] << "\n";
    }
    //std::cout << "likelihood_computation_params: " << treeinfo->likelihood_computation_params << "\n";
    //std::cout << "tree: " << treeinfo->tree << "\n";
    unsigned int branch_count = treeinfo->tree->edge_count;
    std::cout << "linked_branch_lengths:\n";
    for (size_t i = 0; i < branch_count; ++i) {
        std::cout << treeinfo->linked_branch_lengths[i] << "\n";
    }
    std::cout << "param_indices:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->param_indices[i] << "\n";
    }
    std::cout << "params_to_optimize:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->params_to_optimize[i] << "\n";
    }
    std::cout << "partition_loglh:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->partition_loglh[i] << "\n";
    }
    std::cout << "partitions:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->partitions[i] << "\n";
    }
    std::cout << "subst_matrix_symmetries:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->subst_matrix_symmetries[i] << "\n";
    }
    std::cout << "pmatrix_valid:\n";
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->pmatrix_valid[i] << "\n";
    }
    std::cout << "clv_valid:\n";
    assert(treeinfo->clv_valid);
    for (size_t i = 0; i < treeinfo->partition_count; ++i) {
        std::cout << treeinfo->clv_valid[i] << "\n";
    }
}


void printDisplayedTrees(AnnotatedNetwork& ann_network) {
    updateNetwork(ann_network);
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::vector<std::pair<std::string, double>> displayed_trees;
        if (ann_network.network.num_reticulations() == 0) {
            std::string newick = netrax::toExtendedNewick(ann_network);
            displayed_trees.emplace_back(std::make_pair(newick, 1.0));
        } else {
            size_t n_trees = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].num_active_displayed_trees;
            for (size_t j = 0; j < n_trees; ++j) {
                DisplayedTreeData& tree = ann_network.pernode_displayed_tree_data[ann_network.network.root->clv_index].displayed_trees[j];
                pll_utree_t* utree = netrax::displayed_tree_to_utree(ann_network.network, tree.treeLoglData.reticulationChoices.configs[0]);
                double prob = std::exp(tree.treeLoglData.tree_logprob);
                Network displayedNetwork = netrax::convertUtreeToNetwork(*utree, ann_network.options, 0);
                std::string newick = netrax::toExtendedNewick(displayedNetwork);
                pll_utree_destroy(utree, nullptr);
                displayed_trees.emplace_back(std::make_pair(newick, prob));
            }
        }
        std::cout << "Number of displayed trees: " << displayed_trees.size() << "\n";
        std::cout << "Displayed trees Newick strings:\n";
        for (const auto& entry : displayed_trees) {
            std::cout << entry.first << "\n";
        }
        std::cout << "Displayed trees probabilities:\n";
        for (const auto& entry : displayed_trees) {
            std::cout << entry.second << "\n";
        }
    }
}

void printDisplayedTreesChoices(AnnotatedNetwork& ann_network, Node* virtualRoot) {
    NodeDisplayedTreeData& nodeData = ann_network.pernode_displayed_tree_data[virtualRoot->clv_index];
    for (size_t i = 0; i < nodeData.num_active_displayed_trees; ++i) {
        printReticulationChoices(nodeData.displayed_trees[i].treeLoglData.reticulationChoices);
    }
}

void printAllDisplayedTreeConfigs(AnnotatedNetwork& ann_network) {
    for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
        std::cout << "displayed trees at node " << i << ":" << "\n";
        NodeDisplayedTreeData& nodeData = ann_network.pernode_displayed_tree_data[i];
        for (size_t j = 0; j < nodeData.num_active_displayed_trees; ++j) {
            printReticulationChoices(nodeData.displayed_trees[j].treeLoglData.reticulationChoices);
            if (!nodeData.displayed_trees[j].treeLoglData.tree_logprob_valid) {
                nodeData.displayed_trees[j].treeLoglData.tree_logprob = computeReticulationConfigLogProb(nodeData.displayed_trees[j].treeLoglData.reticulationChoices, ann_network.reticulation_probs);
                nodeData.displayed_trees[j].treeLoglData.tree_logprob_valid = true;
            }
            std::cout << " tree prob: " << exp(nodeData.displayed_trees[j].treeLoglData.tree_logprob) << "\n";
        }
    }
}


}
