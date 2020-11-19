#include <iostream>
#include <string>

#include <CLI11.hpp>
#include "Api.hpp"
#include "graph/AnnotatedNetwork.hpp"
#include "NetraxOptions.hpp"
#include "NetworkIO.hpp"

using namespace netrax;

int parseOptions(int argc, char **argv, netrax::NetraxOptions *options) {
    CLI::App app { "NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting" };
    app.add_option("--msa", options->msa_file, "The Multiple Sequence Alignment File");
    app.add_option("-o,--output", options->output_file, "File where to write the final network to");
    app.add_option("--start_network", options->start_network_file, "A network file (in Extended Newick format) to start the search on");
    app.add_option("-r,--reticulations", options->max_reticulations,
            "Maximum number of reticulations to consider (default: 32)");
    app.add_option("-t,--timeout", options->timeout, "Maximum number of seconds to run network search.");
    app.add_flag("-e,--endless", options->endless, "Endless search mode - keep trying with more random start networks.");
    app.add_option("--seed", options->seed, "Seed for random number generation.");
    app.add_flag("--score_only", options->score_only, "Only read a network and MSA from file and compute its score.");
    app.add_flag("--extract_displayed_trees", options->extract_displayed_trees, "Only extract all displayed trees with their probabilities from a network.");
    app.add_flag("--generate_random_network_only", options->generate_random_network_only, "Only generate a random network, with as many reticulations as specified in the -r parameter");
    
    bool best_displayed_tree_variant = false;
    app.add_flag("--best_displayed_tree_variant", best_displayed_tree_variant, "Use only best displayed tree instead of weighted average in network likelihood formula.");

    CLI11_PARSE(app, argc, argv);
    options->likelihood_variant = (best_displayed_tree_variant) ? LikelihoodVariant::BEST_DISPLAYED_TREE : LikelihoodVariant::AVERAGE_DISPLAYED_TREES;
    return 0;
}

void run_random_endless(NetraxOptions& netraxOptions, std::mt19937& rng) {
    double best_score = std::numeric_limits<double>::infinity();
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t start_reticulations = 0;
    while (true) {
        std::cout << "Starting with new random network with " << start_reticulations << " reticulations.\n";
        netrax::AnnotatedNetwork ann_network = NetraxInstance::build_random_annotated_network(netraxOptions);
        NetraxInstance::init_annotated_network(ann_network, rng);
        NetraxInstance::add_extra_reticulations(ann_network, start_reticulations);
        NetraxInstance::optimizeEverything(ann_network);
        double final_bic = NetraxInstance::scoreNetwork(ann_network);
        std::cout << "The inferred network has " << ann_network.network.num_reticulations() << " reticulations and this BIC score: " << final_bic << "\n\n";
        if (final_bic < best_score) {
            best_score = final_bic;
            std::cout << "IMPROVED BEST SCORE FOUND SO FAR: " << best_score << "\n\n";
            NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
            std::cout << "Better network written to " << netraxOptions.output_file << "\n";  
        } else {
            std::cout << "REMAINED BEST SCORE FOUND SO FAR: " << best_score << "\n";
        }
        if (netraxOptions.timeout > 0) {
            auto act_time = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(act_time - start_time).count() >= netraxOptions.timeout) {
                break;
            }
        }
    }
}

void run_random_single(NetraxOptions& netraxOptions, std::mt19937& rng) {
    netrax::AnnotatedNetwork ann_network;
    if (netraxOptions.start_network_file.empty()) {
        ann_network = NetraxInstance::build_random_annotated_network(netraxOptions);
    } else {
        ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    }
    NetraxInstance::init_annotated_network(ann_network, rng);

    NetraxInstance::optimizeEverything(ann_network);
    double final_bic = NetraxInstance::scoreNetwork(ann_network);
    std::cout << "The inferred network has " << ann_network.network.num_reticulations() << " reticulations and this BIC score: " << final_bic << "\n";

    NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
    std::cout << "Final network written to " << netraxOptions.output_file << "\n";
}

void score_only(const NetraxOptions& netraxOptions, std::mt19937& rng) {
    if (netraxOptions.msa_file.empty()) {
        throw std::runtime_error("Need MSA to score a network");
    }
    if (netraxOptions.start_network_file.empty()) {
        throw std::runtime_error("Need network file to be scored");
    }
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);
    NetraxInstance::optimizeModel(ann_network);
    double final_bic = NetraxInstance::scoreNetwork(ann_network);
    double final_logl = NetraxInstance::computeLoglikelihood(ann_network);
    std::cout << "Number of reticulations: " << ann_network.network.num_reticulations() << "\n";
    std::cout << "BIC Score: " << final_bic << "\n";
    std::cout << "Loglikelihood: " << final_logl << "\n";
}

void extract_displayed_trees(const NetraxOptions& netraxOptions, std::mt19937& rng) {
    if (netraxOptions.start_network_file.empty()) {
        throw std::runtime_error("Need network to extract displayed trees");
    }
    std::vector<std::pair<std::string, double> > displayed_trees;
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);
    for (size_t tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index) {
        pll_utree_t* utree = netrax::displayed_tree_to_utree(ann_network.network, tree_index);
        double prob = netrax::displayed_tree_prob(ann_network, tree_index);
        Network displayedNetwork = netrax::convertUtreeToNetwork(*utree, 0);
        std::string newick = netrax::toExtendedNewick(displayedNetwork);
        pll_utree_destroy(utree, nullptr);
        displayed_trees.emplace_back(std::make_pair(newick, prob));
    }

    std::cout << "Number of displayed trees: " << displayed_trees.size() << "\n";
    std::cout << "Displayed trees Newick strings:\n";
    for (const auto& entry : displayed_trees) {
        std::cout << entry.first << "\n";
    }
    std::cout << "Displayed trees probabilities:\n";
    for (const auto& entry : displayed_trees) {
        std::cout << entry.second << "\n";
    }
}

void generate_random_network_only(const NetraxOptions& netraxOptions, std::mt19937& rng) {
    if (netraxOptions.msa_file.empty()) {
        throw std::runtime_error("Need MSA to decide on the number of taxa");
    }
    if (netraxOptions.output_file.empty()) {
        throw std::runtime_error("Need output file to write the generated network");
    }
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_random_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);
    NetraxInstance::add_extra_reticulations(ann_network, netraxOptions.max_reticulations);
    NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
    std::cout << "Final network written to " << netraxOptions.output_file << "\n";
}

int main(int argc, char **argv) {
    //std::ios::sync_with_stdio(false);
    //std::cin.tie(NULL);
    netrax::NetraxOptions netraxOptions;
    parseOptions(argc, argv, &netraxOptions);
    std::mt19937 rng;
    if (netraxOptions.seed == 0) {
        std::random_device dev;
        std::mt19937 rng2(dev());
        rng = rng2;
    } else {
        std::mt19937 rng2(netraxOptions.seed);
        rng = rng2;
    }

    if (netraxOptions.extract_displayed_trees) {
        extract_displayed_trees(netraxOptions, rng);
        return 0;
    }

    if (netraxOptions.generate_random_network_only) {
        generate_random_network_only(netraxOptions, rng);
        return 0;
    }

    std::cout << "The current Likelihood model being used is the DNA model from raxml-ng\n\n";
    if (netraxOptions.msa_file.empty()) {
        throw std::runtime_error("Need MSA to score a network");
    }

    if (netraxOptions.score_only) {
        score_only(netraxOptions, rng);
        return 0;
    } else if (netraxOptions.output_file.empty()) {
        throw std::runtime_error("No output path specified");
    }

    if (!netraxOptions.endless) {
        run_random_single(netraxOptions, rng);
    } else {
        run_random_endless(netraxOptions, rng);
    }

    return 0;
}
