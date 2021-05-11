/*
 * DebugPrintFunctions.hpp
 *
 *  Created on: Jun 4, 2020
 *      Author: sarah
 */

#pragma once

#include <stddef.h>
#include <string>
#include <vector>

#include <raxml-ng/main.hpp>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "moves/MoveType.hpp"

namespace netrax {

struct AnnotatedNetwork;
struct Network;
struct RootedNetwork;
struct Node;

void printClv(const pllmod_treeinfo_t &treeinfo, size_t clv_index, double* clv, size_t partition_index);
void print_clv_vector(pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx, size_t partition_idx,
        size_t clv_index);
void printOperationArray(const std::vector<pll_operation_t> &ops);
void printReticulationParents(Network &network);
void print_brlens(AnnotatedNetwork &ann_network);

void printClvValid(AnnotatedNetwork &ann_network);
void printReticulationFirstParents(AnnotatedNetwork &ann_network);

std::string exportDebugInfoRootedNetwork(const RootedNetwork &rnetwork);
std::string exportDebugInfo(AnnotatedNetwork &ann_network, bool with_labels = true);
std::string exportDebugInfoNetwork(Network &network, bool with_labels = true);

void print_partition(AnnotatedNetwork& ann_network, pll_partition_t* partition);
void print_treeinfo(AnnotatedNetwork& ann_network);
void printDisplayedTrees(AnnotatedNetwork& ann_network);
void printDisplayedTreesChoices(AnnotatedNetwork& ann_network, Node* virtualRoot);

template <typename T>
void printCandidates(std::vector<T>& candidates) {
    if (ParallelContext::master()) {
        std::cout << "The candidates are:\n";
        for (size_t i = 0; i < candidates.size(); ++i) {
            std::cout << toString(candidates[i]) << "\n";
        }
        std::cout << "End of candidates.\n";
    }
}

}
