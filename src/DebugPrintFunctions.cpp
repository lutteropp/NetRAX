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
#include "graph/BiconnectedComponents.hpp"
#include "graph/Direction.hpp"
#include "graph/Edge.hpp"
#include "graph/Link.hpp"
#include "graph/Network.hpp"
#include "graph/NetworkFunctions.hpp"
#include "graph/NetworkTopology.hpp"
#include "graph/Node.hpp"
#include "graph/NodeType.hpp"
#include "io/RootedNetworkParser.hpp"
#include "io/NetworkIO.hpp"
#include "NetraxOptions.hpp"

namespace netrax {

void printClv(const pllmod_treeinfo_t &treeinfo, size_t clv_index, size_t partition_index) {
    size_t sites = treeinfo.partitions[partition_index]->sites;
    size_t rate_cats = treeinfo.partitions[partition_index]->rate_cats;
    size_t states = treeinfo.partitions[partition_index]->states;
    size_t states_padded = treeinfo.partitions[partition_index]->states_padded;
    std::cout << "Clv for clv_index " << clv_index << ": \n";
    for (unsigned int n = 0; n < sites; ++n) {
        for (unsigned int i = 0; i < rate_cats; ++i) {
            for (unsigned int j = 0; j < states; ++j) {
                std::cout
                        << treeinfo.partitions[partition_index]->clv[clv_index][j
                                + i * states_padded] << "\n";
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

void printClvTouched(Network &network, const std::vector<bool> &clv_touched) {
    std::cout << "clv touched:\n";
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        std::cout << "  clv_touched[" << network.nodes[i].clv_index << "]= "
                << clv_touched[network.nodes[i].clv_index] << "\n";
    }
}

void print_dead_nodes(Network &network, const std::vector<bool> &dead_nodes) {
    std::cout << "dead nodes:\n";
    for (size_t i = 0; i < network.num_nodes(); ++i) {
        std::cout << "  dead_nodes[" << network.nodes[i].clv_index << "]= "
                << dead_nodes[network.nodes[i].clv_index] << "\n";
    }
}

void print_brlens(AnnotatedNetwork &ann_network) {
    std::cout << "brlens:\n";
    size_t n_partitions = 1;
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        n_partitions = ann_network.fake_treeinfo->partition_count;
    }
    for (size_t p = 0; p < n_partitions; ++p) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            std::cout << "brlens[" << p << "][" << ann_network.network.edges[i].pmatrix_index
                    << "]: " << ann_network.network.edges[i].length << "\n";
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

void printReticulationNodesPerMegablob(AnnotatedNetwork &ann_network) {
    std::cout << "reticulation nodes per megablob:\n";
    for (size_t i = 0; i < ann_network.blobInfo.megablob_roots.size(); ++i) {
        std::cout << "reticulation nodes in megablob "
                << ann_network.blobInfo.megablob_roots[i]->clv_index << ":\n";
        for (size_t j = 0; j < ann_network.blobInfo.reticulation_nodes_per_megablob[i].size();
                ++j) {
            std::cout << "  "
                    << ann_network.blobInfo.reticulation_nodes_per_megablob[i][j]->clv_index
                    << "\n";
        }
    }
    std::cout << '\n';
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

std::string buildNodeGraphics(const Node *node, const BlobInformation &blobInfo) {
    std::stringstream ss;
    ss << "\t\tgraphics\n\t\t[\n";
    ss << "\t\t\tfill\t\"";
    if (node->type == NodeType::RETICULATION_NODE) {
        ss << "#00CCFF";
    } else if (std::find(blobInfo.megablob_roots.begin(), blobInfo.megablob_roots.end(), node)
            != blobInfo.megablob_roots.end()) { // megablob root
        ss << "#FF6600";
    } else { // normal node
        ss << "#FFCC00";
    }
    ss << "\"\n\t\t]\n";
    return ss.str();
}

std::string exportDebugInfoBlobs(Network &network, const BlobInformation &blobInfo) {
    std::stringstream ss;
    ss << "graph\n[\tdirected\t1\n";
    std::vector<Node*> parent = grab_current_node_parents(network);
    for (size_t i = 0; i < network.nodes.size(); ++i) {
        if (network.nodes_by_index[i] == nullptr) {
            continue;
        }
        ss << "\tnode\n\t[\n\t\tid\t" << i << "\n";
        std::string nodeLabel = std::to_string(i);
        if (!network.nodes_by_index[i]->label.empty()) {
            nodeLabel += ": " + network.nodes_by_index[i]->label;
        }
        nodeLabel += "|" + std::to_string(blobInfo.node_blob_id[i]);

        ss << "\t\tlabel\t\"" << nodeLabel << "\"\n";
        ss << buildNodeGraphics(network.nodes_by_index[i], blobInfo);
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
        std::string edgeLabel = std::to_string(i) + "|" + std::to_string(blobInfo.edge_blob_id[i]);
        ss << "\"" << edgeLabel << "\"\n";
        ss << "\t]\n";
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
    updateNetwork(ann_network);
    return exportDebugInfoExtraNodeNumber(ann_network.network, std::vector<unsigned int>(), with_label);
}

std::string exportDebugInfoNetwork(Network &network, bool with_labels) {
    return exportDebugInfoExtraNodeNumber(network, std::vector<unsigned int>(), with_labels);
}

}
