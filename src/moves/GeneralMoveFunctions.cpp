#include "GeneralMoveFunctions.hpp"

#include "../helper/Helper.hpp"
#include "../DebugPrintFunctions.hpp"
#include "Move.hpp"

namespace netrax {

size_t getRandomIndex(std::mt19937& rng, size_t n) {
    std::uniform_int_distribution<std::mt19937::result_type> d(0, n-1);
    return d(rng);
}

Edge* getRandomEdge(AnnotatedNetwork &ann_network) {
    size_t n = ann_network.network.num_branches();
    return ann_network.network.edges_by_index[getRandomIndex(ann_network.rng, n)];
}

Node* addInnerNode(AnnotatedNetwork &ann_network, ReticulationData *retData, size_t wanted_clv_index) {
    Network& network = ann_network.network;
    assert(network.num_nodes() < network.nodes.size());
    unsigned int clv_index;

    if (wanted_clv_index < network.nodes.size()) {
        if (network.nodes_by_index[wanted_clv_index] != nullptr)  {
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << exportDebugInfo(ann_network) << "\n";
            }
            throw std::runtime_error("wanted clv index " + std::to_string(wanted_clv_index) + " is already taken");
        }
        clv_index = wanted_clv_index;
    } else {
        clv_index = network.nodes.size() - 1;
        // try to find a smaller unused clv index
        for (size_t i = 0; i < clv_index; ++i) {
            if (network.nodes_by_index[i] == nullptr) {
                clv_index = i;
                break;
            }
        }
    }
    assert(network.nodes_by_index[clv_index] == nullptr);
    unsigned int scaler_index = clv_index - network.num_tips();

    // find an empty place in the edges array
    size_t index_in_nodes_array = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < network.nodes.size(); ++i) {
        if (network.nodes[i].clv_index == std::numeric_limits<size_t>::max()) {
            index_in_nodes_array = i;
            break;
        }
    }
    assert(index_in_nodes_array < std::numeric_limits<size_t>::max());


    network.nodes_by_index[clv_index] = &network.nodes[index_in_nodes_array];

    if (retData) {
        network.nodes[index_in_nodes_array].initReticulation(clv_index, scaler_index, "", *retData);
        network.reticulation_nodes.emplace_back(network.nodes_by_index[clv_index]);
        network.nodes[index_in_nodes_array].getReticulationData()->reticulation_index =
                network.reticulation_nodes.size() - 1;
        for (size_t i = 0; i < network.reticulation_nodes.size(); ++i) {
            assert(network.reticulation_nodes[i]->type == NodeType::RETICULATION_NODE);
        }
    } else {
        network.nodes[index_in_nodes_array].initBasic(clv_index, scaler_index, "");
    }

    network.nodeCount++;
    return network.nodes_by_index[clv_index];
}

Edge* addEdgeInternal(AnnotatedNetwork &ann_network, Link *link1, Link *link2, double length,
        size_t pmatrix_index) {
    assert(ann_network.network.num_branches() < ann_network.network.edges.size());
    if (link1->direction == Direction::INCOMING) {
        std::swap(link1, link2);
    }

    assert(ann_network.network.edges_by_index[pmatrix_index] == nullptr);

    // find an empty place in the edges array
    size_t index_in_edges_array = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < ann_network.network.edges.size(); ++i) {
        if (ann_network.network.edges[i].pmatrix_index == std::numeric_limits<size_t>::max()) {
            index_in_edges_array = i;
            break;
        }
    }
    assert(index_in_edges_array < std::numeric_limits<size_t>::max());

    ann_network.network.edges[index_in_edges_array].init(pmatrix_index, link1, link2, length, 1.0);
    ann_network.network.edges_by_index[pmatrix_index] = &ann_network.network.edges[index_in_edges_array];
    ann_network.network.branchCount++;

    return ann_network.network.edges_by_index[pmatrix_index];
}

Edge* addEdge(AnnotatedNetwork &ann_network, Link *link1, Link *link2, double length,
        size_t wanted_pmatrix_index) {
    if (link1->direction == Direction::INCOMING) {
        std::swap(link1, link2);
    }

    size_t pmatrix_index = 0;
    if (wanted_pmatrix_index < ann_network.network.edges.size()) {
        if (ann_network.network.edges_by_index[wanted_pmatrix_index] != nullptr)  {
            if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
                std::cout << exportDebugInfo(ann_network) << "\n";
            }
            throw std::runtime_error("wanted pmatrix index " + std::to_string(wanted_pmatrix_index) + " is already taken");
        }
        pmatrix_index = wanted_pmatrix_index;
    } else {
        // find smallest free non-tip pmatrix index
        for (size_t i = 0; i < ann_network.network.edges.size(); ++i) {
            if (ann_network.network.edges_by_index[i] == nullptr) {
                pmatrix_index = i;
                break;
            }
        }
    }
    assert(ann_network.network.edges_by_index[pmatrix_index] == nullptr);

    return addEdgeInternal(ann_network, link1, link2, length, pmatrix_index);
}

std::vector<size_t> determineEdgeOrder(AnnotatedNetwork& ann_network, size_t start_edge_idx) {
    std::vector<size_t> res(ann_network.network.num_branches(), std::numeric_limits<size_t>::infinity());
    res[start_edge_idx] = 0;
    std::queue<Edge*> q;
    q.emplace(ann_network.network.edges_by_index[start_edge_idx]);
    while (!q.empty()) {
        Edge* act_edge = q.front();
        q.pop();
        std::unordered_set<size_t> neigh_indices = getNeighborPmatrixIndices(ann_network.network, act_edge);
        for (size_t neigh_idx : neigh_indices) {
            if (res[neigh_idx] > res[act_edge->pmatrix_index] + 1) {
                res[neigh_idx] = res[act_edge->pmatrix_index] + 1;
                q.emplace(ann_network.network.edges_by_index[neigh_idx]);
            }
        }
    }
    return res;
}

std::vector<double> get_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index) {
    std::vector<double> res(ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED ? ann_network.fake_treeinfo->partition_count : 1, 0.0);
    if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            res[p] = ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index];
        }
    } else {
        res[0] = ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index];
        assert(res[0] >= ann_network.options.brlen_min);
        assert(res[0] <= ann_network.options.brlen_max);
    }
    return res;
}

std::vector<double> get_halved_edge_lengths(const std::vector<double>& lengths, double min_br) {
    std::vector<double> res(lengths.size());
    for (size_t p = 0; p < lengths.size(); ++p) {
        res[p] = std::max(lengths[p] / 2, min_br);
    }
    return res;
}

std::vector<double> get_minus_edge_lengths(const std::vector<double>& lengths1, const std::vector<double>& lengths2, double min_br) {
    assert(lengths1.size() == lengths2.size());
    std::vector<double> res(lengths1.size());
    for (size_t p = 0; p < lengths1.size(); ++p) {
        res[p] = std::max(lengths1[p] - lengths2[p], min_br);
    }
    return res;
}

std::vector<double> get_plus_edge_lengths(const std::vector<double>& lengths1, const std::vector<double>& lengths2, double max_br) {
    assert(lengths1.size() == lengths2.size());
    std::vector<double> res(lengths1.size());
    for (size_t p = 0; p < lengths1.size(); ++p) {
        res[p] = std::min(lengths1[p] + lengths2[p], max_br);
    }
    return res;
}

void set_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index, const std::vector<double> &lengths) {
    if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        for (size_t p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            // skip remote partitions
            if (!ann_network.fake_treeinfo->partitions[p]) {
                continue;
            }
            ann_network.fake_treeinfo->branch_lengths[p][pmatrix_index] = lengths[p];
            assert(lengths[p] >= ann_network.options.brlen_min);
            assert(lengths[p] <= ann_network.options.brlen_max);
        }
    } else {
        ann_network.fake_treeinfo->linked_branch_lengths[pmatrix_index] = lengths[0];
        ann_network.network.edges_by_index[pmatrix_index]->length = lengths[0];
        assert(lengths[0] >= ann_network.options.brlen_min);
        assert(lengths[0] <= ann_network.options.brlen_max);
    }
}

bool hasPath(Network &network, const Node *from, const Node *to, bool nonelementary) {
    std::vector<bool> visited(network.num_nodes(), false);
    std::queue<std::pair<const Node*, const Node*> > q;
    q.emplace(to, nullptr);
    while (!q.empty()) {
        const Node *node = q.front().first;
        const Node *child = q.front().second;
        if (node == from) {
            if (!nonelementary || child != to) {
                return true;
            }
        }
        q.pop();
        visited[node->clv_index] = true;
        for (const Node *neigh : getAllParents(network, node)) {
            if (!visited[neigh->clv_index] || (nonelementary && neigh == from)) {
                q.emplace(std::make_pair(neigh, node));
            }
        }
    }
    return false;
}

void swapPmatrixIndex(AnnotatedNetwork& ann_network, Move& move, size_t old_pmatrix_index, size_t new_pmatrix_index, bool undo) {
    if (old_pmatrix_index == new_pmatrix_index) {
        return;
    }
    // update pmatrix valid and the pmatrices
    for (unsigned int p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        if (!ann_network.fake_treeinfo->partitions[p]) { // skip remote partitions
            continue;
        }
        std::swap(ann_network.fake_treeinfo->pmatrix_valid[p][old_pmatrix_index], ann_network.fake_treeinfo->pmatrix_valid[p][new_pmatrix_index]);
        std::swap(ann_network.fake_treeinfo->partitions[p]->pmatrix[old_pmatrix_index], ann_network.fake_treeinfo->partitions[p]->pmatrix[new_pmatrix_index]);
    }
    // update branch lengths array
    std::swap(ann_network.fake_treeinfo->linked_branch_lengths[old_pmatrix_index], ann_network.fake_treeinfo->linked_branch_lengths[new_pmatrix_index]);
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        for (unsigned int p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
            if (!ann_network.fake_treeinfo->partitions[p]) { // skip remote partitions
                continue;
            }
            std::swap(ann_network.fake_treeinfo->branch_lengths[p][old_pmatrix_index], ann_network.fake_treeinfo->branch_lengths[p][new_pmatrix_index]);
        }
    }
    // update pmatrix index stored in the edge
    Edge* edge = ann_network.network.edges_by_index[old_pmatrix_index];
    edge->pmatrix_index = new_pmatrix_index;
    // update pmatrix index stored in the links belonging to the edge
    edge->link1->edge_pmatrix_index = new_pmatrix_index;
    edge->link1->outer->edge_pmatrix_index = new_pmatrix_index;
    edge->link2->edge_pmatrix_index = new_pmatrix_index;
    edge->link2->outer->edge_pmatrix_index = new_pmatrix_index;
    // update edges_by_index array
    std::swap(ann_network.network.edges_by_index[old_pmatrix_index], ann_network.network.edges_by_index[new_pmatrix_index]);
    // update pmatrix indices in the move
    updateMovePmatrixIndex(move, old_pmatrix_index, new_pmatrix_index, undo);
}

void swapClvIndex(AnnotatedNetwork& ann_network, Move& move, size_t old_clv_index, size_t new_clv_index, bool undo) {
    if (old_clv_index == new_clv_index) {
        return;
    }
    Node* node = ann_network.network.nodes_by_index[old_clv_index];
    assert(node);

    int old_scaler_index = node->isTip() ? -1 : (old_clv_index - ann_network.network.num_tips());
    int new_scaler_index = node->isTip() ? -1 : (new_clv_index - ann_network.network.num_tips());

    // update clv valid and the clvs
    for (unsigned int p = 0; p < ann_network.fake_treeinfo->partition_count; ++p) {
        if (!ann_network.fake_treeinfo->partitions[p]) { // skip remote partitions
            continue;
        }
        std::swap(ann_network.fake_treeinfo->clv_valid[p][old_clv_index], ann_network.fake_treeinfo->clv_valid[p][new_clv_index]);
        std::swap(ann_network.fake_treeinfo->partitions[p]->clv[old_clv_index], ann_network.fake_treeinfo->partitions[p]->clv[new_clv_index]);
    }

    // update brlen scalers
    if (ann_network.options.brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED && !node->isTip()) {
        std::swap(ann_network.fake_treeinfo->brlen_scalers[old_scaler_index], ann_network.fake_treeinfo->brlen_scalers[new_scaler_index]);
    }

    // update displayed tree data
    std::swap(ann_network.pernode_displayed_tree_data[old_clv_index], ann_network.pernode_displayed_tree_data[new_clv_index]);
    // update clv index stored in the node
    node->clv_index = new_clv_index;
    // update scaler index stored in the node
    node->scaler_index = new_clv_index - ann_network.network.num_tips();
    // update clv index in the links belonging to the node
    for (Link& link : node->links) {
        link.node_clv_index = new_clv_index;
    }
    // update nodes_by_index array
    std::swap(ann_network.network.nodes_by_index[old_clv_index], ann_network.network.nodes_by_index[new_clv_index]);

    // update clv indices in the move
    updateMoveClvIndex(move, old_clv_index, new_clv_index, undo);
}

void swapReticulationIndex(AnnotatedNetwork& ann_network, Move& move, size_t old_reticulation_index, size_t new_reticulation_index, bool undo) {
    if (old_reticulation_index == new_reticulation_index) {
        return;
    }
    // update reticulation states in the displayed trees data
    for (size_t i = 0; i < ann_network.pernode_displayed_tree_data.size(); ++i) {
        for (size_t j = 0; j < ann_network.pernode_displayed_tree_data[i].num_active_displayed_trees; ++j) {
            ReticulationConfigSet& rcs = ann_network.pernode_displayed_tree_data[i].displayed_trees[j].treeLoglData.reticulationChoices;
            for (size_t k = 0; k < rcs.configs.size(); ++k) {
                std::swap(rcs.configs[k][old_reticulation_index], rcs.configs[k][new_reticulation_index]);
            }
        }
    }
    Node* node = ann_network.network.reticulation_nodes[old_reticulation_index];
    assert(node->getType() == NodeType::RETICULATION_NODE);
    // update reticulation index stored in the node
    node->getReticulationData()->reticulation_index = new_reticulation_index;
    // update reticulation nodes array
    std::swap(ann_network.network.reticulation_nodes[old_reticulation_index], ann_network.network.reticulation_nodes[new_reticulation_index]);
    // update reticulation probs array
    std::swap(ann_network.reticulation_probs[old_reticulation_index], ann_network.reticulation_probs[new_reticulation_index]);

    // update move data
    if (!undo) {
        move.remapped_reticulation_indices.emplace_back(std::make_pair(old_reticulation_index, new_reticulation_index));
    }
}

void removeNode(AnnotatedNetwork &ann_network, Move& move, Node *node, bool undo) {
    assert(node);
    assert(!node->isTip());
    Network& network = ann_network.network;

    // move the node the the last index, the remove the node.
    if (node->type == NodeType::RETICULATION_NODE) {
        swapReticulationIndex(ann_network, move, node->getReticulationData()->reticulation_index, ann_network.network.num_reticulations() - 1, undo);
        network.reticulation_nodes.resize(network.reticulation_nodes.size() - 1);
    }
    swapClvIndex(ann_network, move, node->clv_index, ann_network.network.num_nodes() - 1);
    network.nodes_by_index[node->clv_index] = nullptr;
    node->clear();
    network.nodeCount--;
}

void removeEdge(AnnotatedNetwork &ann_network, Move& move, Edge *edge, bool undo) {
    assert(edge);
    Network& network = ann_network.network;

    // move the node the the last index, the remove the node.
    swapPmatrixIndex(ann_network, move, edge->pmatrix_index, ann_network.network.num_branches() - 1, undo);
    ann_network.network.edges_by_index[edge->pmatrix_index] = nullptr;
    edge->clear();
    ann_network.network.branchCount--;
}

void fixReticulationLinks(AnnotatedNetwork& ann_network) {
    for (size_t i = 0; i < ann_network.network.reticulation_nodes.size(); ++i) {
        ReticulationData* retData = ann_network.network.reticulation_nodes[i]->getReticulationData().get();
        if (retData->link_to_first_parent->edge_pmatrix_index
            > retData->link_to_second_parent->edge_pmatrix_index) {
            std::swap(retData->link_to_first_parent, retData->link_to_second_parent);
        }
    }
}

}