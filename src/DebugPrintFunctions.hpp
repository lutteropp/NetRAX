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

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

struct AnnotatedNetwork;
struct Network;
struct RootedNetwork;
struct BlobInformation;

void printClv(const pllmod_treeinfo_t &treeinfo, size_t clv_index, size_t partition_index);
void print_clv_vector(pllmod_treeinfo_t &fake_treeinfo, size_t tree_idx, size_t partition_idx, size_t clv_index);
void printOperationArray(const std::vector<pll_operation_t> &ops);
void printReticulationParents(Network &network);
void printClvTouched(Network &network, const std::vector<bool> &clv_touched);
void print_dead_nodes(Network &network, const std::vector<bool> &dead_nodes);
void print_brlens(AnnotatedNetwork &ann_network);

void printClvValid(AnnotatedNetwork &ann_network);
void printReticulationFirstParents(AnnotatedNetwork &ann_network);
void printReticulationNodesPerMegablob(AnnotatedNetwork &ann_network);

std::string exportDebugInfoRootedNetwork(const RootedNetwork &rnetwork);
std::string exportDebugInfoBlobs(Network &network, const BlobInformation &blobInfo);
std::string exportDebugInfo(Network &network);

}
