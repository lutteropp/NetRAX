#include <iostream>
#include <string>

#include <CLI11.hpp>
#include "Api.hpp"
#include "graph/AnnotatedNetwork.hpp"
#include "NetraxOptions.hpp"

using namespace netrax;

int parseOptions(int argc, char **argv, netrax::NetraxOptions *options) {
    CLI::App app { "NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting" };
    app.add_option("--msa", options->msa_file, "The Multiple Sequence Alignment File")->required();
    app.add_option("--network", options->network_file, "The Network File")->required();
    app.add_option("-r,--reticulations", options->max_reticulations,
            "Maximum number of reticulations to consider (default: 20)");
    app.add_option("-o,--output", options->output_file, "File where to write the final network to");
    app.add_option("-t,--timeout", options->timeout, "Maximum number of seconds to run network search.");
    CLI11_PARSE(app, argc, argv);
    return 0;
}

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    netrax::NetraxOptions netraxOptions;
    parseOptions(argc, argv, &netraxOptions);

    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    std::cout << "The current Likelihood model being used is the DNA model from raxml-ng\n\n";

    double final_bic = NetraxInstance::optimizeEverything(ann_network);
    std::cout << "The inferred network has " << ann_network.network.num_reticulations() << " reticulations and this BIC score: " << final_bic << "\n";

    if (!netraxOptions.output_file.empty()) {
        NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
        std::cout << "Final network written to " << netraxOptions.output_file << "\n";
    }
    return 0;
}
