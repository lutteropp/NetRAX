/*
 * Api.hpp
 *
 *  Created on: Apr 24, 2020
 *      Author: sarah
 */

#pragma once

#include <string>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

namespace netrax {

struct AnnotatedNetwork;
class NetraxOptions;


struct NetraxInstance {
// only put these into a struct in order to get nicer doxygen documentation
        static AnnotatedNetwork build_annotated_network(const NetraxOptions &options);
        static AnnotatedNetwork build_annotated_network_from_string(const NetraxOptions &options,
                const std::string &newickString);
        static AnnotatedNetwork build_annotated_network_from_utree(const NetraxOptions &options,
                const pll_utree_t &utree);
        static AnnotatedNetwork build_random_annotated_network(const NetraxOptions &options,
                unsigned int start_reticulations = 0);
        static AnnotatedNetwork build_parsimony_annotated_network(const NetraxOptions &options,
                unsigned int start_reticulations = 0);
        static AnnotatedNetwork build_best_raxml_annotated_network(const NetraxOptions &options,
                unsigned int start_reticulations = 0);
        static double computeLoglikelihood(AnnotatedNetwork &ann_network);
        static double scoreNetwork(AnnotatedNetwork &ann_network);
        static void updateReticulationProbs(AnnotatedNetwork &ann_network);
        static void optimizeModel(AnnotatedNetwork &ann_network);
        static void optimizeBranches(AnnotatedNetwork &ann_network);
        static void optimizeTopology(AnnotatedNetwork &ann_network);
        static void optimizeEverything(AnnotatedNetwork &ann_network);
        static void writeNetwork(AnnotatedNetwork &ann_network, const std::string &filepath);
};

}
