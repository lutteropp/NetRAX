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

int parseOptions(int argc, char** argv, netrax::NetraxOptions* options) {
	CLI::App app { "NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting" };
	app.add_option("--msa", options->msa_file, "The Multiple Sequence Alignment File")->required();
	app.add_option("--network", options->network_file, "The Network File")->required();
	//app.add_option("-r,--reticulations", options->num_reticulations, "Number of reticulations");
	CLI11_PARSE(app, argc, argv);
	return 0;
}

int main(int argc, char** argv) {
	std::ios::sync_with_stdio(false);
	std::cin.tie(NULL);

	netrax::NetraxOptions netraxOptions;
	parseOptions(argc, argv, &netraxOptions);

	netrax::Network network = netrax::readNetworkFromFile(netraxOptions.network_file);
	netrax::RaxmlWrapper wrapper(netraxOptions);
	Options raxmlOptions = wrapper.getRaxmlOptions();
	TreeInfo treeinfo = wrapper.createRaxmlTreeinfo(network);
	netrax::RaxmlWrapper::NetworkParams* params = (netrax::RaxmlWrapper::NetworkParams*) treeinfo.pll_treeinfo().likelihood_computation_params;
	pllmod_treeinfo_t* fake_treeinfo = params->network_treeinfo;

	std::cout << "The current Likelihood model being used is the DNA model from raxml-ng\n\n";
	std::cout << "Initial network loglikelihood: " << treeinfo.loglh(false) << "\n";
	std::cout << "After updating reticulation probs: " << netrax::computeLoglikelihood(network, *fake_treeinfo, 0, 1, true) << "\n";
	std::cout << "After doing model optimization: " << treeinfo.optimize_model(raxmlOptions.lh_epsilon) << "\n";
	std::cout << "After updating reticulation probs again: " << netrax::computeLoglikelihood(network, *fake_treeinfo, 0, 1, true) << "\n";
	std::cout << "After branch length optimization (by averaging over per-tree optimized brlens in easy cases): " << treeinfo.optimize_branches(raxmlOptions.lh_epsilon, 1) << "\n";
	std::cout << "After updating reticulation probs again again: " << netrax::computeLoglikelihood(network, *fake_treeinfo, 0, 1, true) << "\n";
	std::cout << "(Topology optimization not implemented yet) \n";
	return 0;
}
