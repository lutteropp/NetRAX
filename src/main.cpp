#include <libpll/pll_tree.h>
#include <PartitionedMSA.hpp>
#include <CLI11.hpp>

#include <iostream>
#include <string>

#include "Options.hpp"
#include "Network.hpp"
#include "io/NetworkIO.hpp"

#include "likelihood/LikelihoodComputation.hpp"
#include "optimization/BranchLengthOptimization.hpp"
#include "optimization/ModelOptimization.hpp"
#include "traversal/Traversal.hpp"

int parseOptions(int argc, char** argv, netrax::Options* options) {
	CLI::App app { "NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting" };
	app.add_option("--msa", options->msa_file, "The Multiple Sequence Alignment File")->required();
	app.add_option("--network", options->network_file, "The Network File");
	app.add_option("-r,--reticulations", options->num_reticulations, "Number of reticulations");
	CLI11_PARSE(app, argc, argv);
	return 0;
}

int main(int argc, char** argv) {
	netrax::Options options;
	parseOptions(argc, argv, &options);
	return 0;
}

