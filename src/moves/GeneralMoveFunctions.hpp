#pragma once

#include "MoveType.hpp"
#include <vector>
#include <unordered_set>

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

size_t getRandomIndex(std::mt19937& rng, size_t n);
Edge* getRandomEdge(AnnotatedNetwork &ann_network);

Node* addInnerNode(AnnotatedNetwork &ann_network, ReticulationData *retData, size_t wanted_clv_index);

Edge* addEdge(AnnotatedNetwork &ann_network, Link *link1, Link *link2, double length,
        size_t wanted_pmatrix_index);
std::vector<size_t> determineEdgeOrder(AnnotatedNetwork& ann_network, size_t start_edge_idx);
void resetReticulationLinks(Node *node);
void addRepairCandidates(Network &network, std::unordered_set<Node*> &repair_candidates,
        Node *node);
std::vector<double> get_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index);
std::vector<double> get_halved_edge_lengths(const std::vector<double>& lengths, double min_br);
std::vector<double> get_minus_edge_lengths(const std::vector<double>& lengths1, const std::vector<double>& lengths2, double min_br);
std::vector<double> get_plus_edge_lengths(const std::vector<double>& lengths1, const std::vector<double>& lengths2, double max_br);
void set_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index, const std::vector<double> &lengths);
bool hasPath(Network &network, const Node *from, const Node *to, bool nonelementary = false);

template<typename T>
void sortByProximity(std::vector<T>& candidates, AnnotatedNetwork& ann_network) {
    if (!ann_network.options.reorder_candidates) {
        return;
    }
    size_t start_edge_idx = ann_network.last_accepted_move_edge_orig_idx;
    if (start_edge_idx >= ann_network.network.num_branches()) {
        return;
    }
    std::vector<size_t> edge_order = determineEdgeOrder(ann_network, start_edge_idx);
    std::sort(candidates.begin(), candidates.end(), [&edge_order](const T& a, const T& b) {
        return edge_order[a.edge_orig_idx] < edge_order[b.edge_orig_idx];
    });
}

std::vector<double> get_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index);

void swapPmatrixIndex(AnnotatedNetwork& ann_network, Move& move, size_t old_pmatrix_index, size_t new_pmatrix_index, bool undo = false);
void swapClvIndex(AnnotatedNetwork& ann_network, Move& move, size_t old_clv_index, size_t new_clv_index, bool undo = false);
void swapReticulationIndex(AnnotatedNetwork& ann_network, Move& move, size_t old_reticulation_index, size_t new_reticulation_index, bool undo = false);

void removeNode(AnnotatedNetwork &ann_network, Move& move, Node *node, bool undo);
void removeEdge(AnnotatedNetwork &ann_network, Move& move, Edge *edge, bool undo);

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i+1 < vec.size()) {
            os << ", ";
        }
    }
    return os;
}

}