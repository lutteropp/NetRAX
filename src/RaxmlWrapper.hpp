/*
 * RaxmlWrapper.hpp
 *
 *  Created on: Sep 28, 2019
 *      Author: sarah
 */

#pragma once

#include <raxml-ng/main.hpp>
#include <raxml-ng/Model.hpp>

#include "NetraxOptions.hpp"
#include "graph/Network.hpp"
#include "graph/AnnotatedNetwork.hpp"
namespace netrax {

struct NetworkParams {
        AnnotatedNetwork *ann_network;
        NetworkParams(AnnotatedNetwork *ann_network) :
                ann_network(ann_network) {
        }
};

RaxmlInstance createRaxmlInstance(const NetraxOptions& options);
void enableRaxmlDebugOutput(RaxmlInstance& instance);

Tree generateRandomTree(const RaxmlInstance& instance, double seed);
Tree generateParsimonyTree(const RaxmlInstance& instance, double seed);
Tree bestRaxmlTree(const RaxmlInstance& instance);
pllmod_treeinfo_t* createNetworkPllTreeinfo(AnnotatedNetwork &ann_network);

TreeInfo* createRaxmlTreeinfo(AnnotatedNetwork &ann_network); // Creates a network treeinfo
TreeInfo* createRaxmlTreeinfo(pll_utree_t *utree, const RaxmlInstance& instance); // Creates a tree treeinfo
TreeInfo* createRaxmlTreeinfo(pll_utree_t *utree, const RaxmlInstance& instance, const pllmod_treeinfo_t &model_treeinfo); // Creates a tree treeinfo, taking the model from the given treeinfo

}
