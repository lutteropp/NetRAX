#include <libpll/pll_tree.h>
#include <CLI11.hpp>

#include <iostream>
#include <string>

#include <raxml-ng/TreeInfo.hpp>
#include "NetraxOptions.hpp"
#include "RaxmlWrapper.hpp"
#include "graph/Network.hpp"
#include "io/NetworkIO.hpp"
#include "likelihood/LikelihoodComputation.hpp"

#include "NetworkInfo.hpp"

int parseOptions(int argc, char **argv, netrax::NetraxOptions *options) {
    CLI::App app { "NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting" };
    app.add_option("--msa", options->msa_file, "The Multiple Sequence Alignment File")->required();
    app.add_option("--network", options->network_file, "The Network File")->required();
    //app.add_option("-r,--reticulations", options->num_reticulations, "Number of reticulations");
    CLI11_PARSE(app, argc, argv);
    return 0;
}

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    netrax::NetraxOptions netraxOptions;
    parseOptions(argc, argv, &netraxOptions);

    netrax::NetworkInfo networkinfo = buildNetworkInfo(netraxOptions);

    std::cout << "The current Likelihood model being used is the DNA model from raxml-ng\n\n";
    std::cout << "Initial network loglikelihood: " << loglh(networkinfo, false) << "\n";
    std::cout << "After updating reticulation probs: " << update_reticulation_probs(networkinfo) << "\n";
    std::cout << "After doing model optimization: " << optimize_model(networkinfo) << "\n";
    std::cout << "After updating reticulation probs again: " << update_reticulation_probs(networkinfo) << "\n";
    std::cout << "After branch length optimization (by averaging over per-tree optimized brlens in easy cases): "
            << optimize_branches(networkinfo) << "\n";
    std::cout << "After updating reticulation probs again again: " << update_reticulation_probs(networkinfo) << "\n";
    std::cout << "(Topology optimization not implemented yet) \n";
    return 0;
}
