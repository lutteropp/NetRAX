#include <iostream>
#include <string>

#include <CLI11.hpp>
#include <mpreal.h>
#include "Api.hpp"
#include "graph/AnnotatedNetwork.hpp"
#include "NetraxOptions.hpp"
#include "NetworkIO.hpp"
#include "DebugPrintFunctions.hpp"

using namespace netrax;

int parseOptions(int argc, char **argv, netrax::NetraxOptions *options) {
    CLI::App app { "NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting" };
    app.add_option("--msa", options->msa_file, "The Multiple Sequence Alignment File");
    app.add_option("--model", options->model_file, "The partitions assignment in case of a partitioned MSA.");
    app.add_option("-o,--output", options->output_file, "File where to write the final network to");
    app.add_option("--start_network", options->start_network_file, "A network file (in Extended Newick format) to start the search on");
    app.add_option("-r,--reticulations", options->max_reticulations,
            "Maximum number of reticulations to consider (default: 32)");
    app.add_option("-n,--num_random_start_networks", options->num_random_start_networks,
            "Number of random start networks (default: 10)");
    app.add_option("-p,--num_parsimony_start_networks", options->num_parsimony_start_networks,
            "Number of parsimony start networks (default: 10)");
    app.add_option("-t,--timeout", options->timeout, "Maximum number of seconds to run network search.");
    app.add_flag("-e,--endless", options->endless, "Endless search mode - keep trying with more random start networks.");
    app.add_option("--seed", options->seed, "Seed for random number generation.");
    app.add_flag("--score_only", options->score_only, "Only read a network and MSA from file and compute its score.");
    app.add_flag("--extract_displayed_trees", options->extract_displayed_trees, "Only extract all displayed trees with their probabilities from a network.");
    app.add_flag("--extract_taxon_names", options->extract_taxon_names, "Only extract all taxon names from a network.");
    app.add_flag("--generate_random_network_only", options->generate_random_network_only, "Only generate a random network, with as many reticulations as specified in the -r parameter");
    app.add_flag("--pretty_print_only", options->pretty_print_only, "Only pretty-print a given input network.");

    std::string brlen_linkage = "scaled";
    app.add_option("--brlen", brlen_linkage, "branch length linkage between partitions (linked, scaled, or unlinked) (default: scaled)");

    bool best_displayed_tree_variant = false;
    app.add_flag("--best_displayed_tree_variant", best_displayed_tree_variant, "Use only best displayed tree instead of weighted average in network likelihood formula.");

    CLI11_PARSE(app, argc, argv);
    options->likelihood_variant = (best_displayed_tree_variant) ? LikelihoodVariant::BEST_DISPLAYED_TREE : LikelihoodVariant::AVERAGE_DISPLAYED_TREES;

    if (brlen_linkage == "scaled") {
        options->brlen_linkage = PLLMOD_COMMON_BRLEN_SCALED;
    } else if (brlen_linkage == "linked") {
        options->brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
    } else if (brlen_linkage == "unlinked") {
        options->brlen_linkage = PLLMOD_COMMON_BRLEN_UNLINKED;
    } else {
        throw std::runtime_error("brlen_linkage needs to be one of {linked, scaled, unlinked}");
    }
    return 0;
}

void run_single_start_waves(NetraxOptions& netraxOptions, std::mt19937& rng) {
    std::vector<MoveType> typesBySpeed = {MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::TailMove, MoveType::HeadMove};

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_score = std::numeric_limits<double>::infinity();
    int best_num_reticulations = 0;
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);

    std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";
    std::string best_network = toExtendedNewick(ann_network);

    bool seen_improvement = true;
    while (seen_improvement) {
        seen_improvement = false;

        double new_score = NetraxInstance::optimizeEverythingRun(ann_network, typesBySpeed, start_time);
        std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
        std::cout << "Best optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";

        if (new_score < best_score) {
            best_score = new_score;
            best_num_reticulations = ann_network.network.num_reticulations();
            std::cout << "IMPROVED BEST SCORE FOUND SO FAR: " << best_score << "\n\n";
            NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
            best_network = toExtendedNewick(ann_network);
            std::cout << best_network << "\n";
            std::cout << "Better network written to " << netraxOptions.output_file << "\n";

            //print displayed trees
            std::vector<std::pair<std::string, double>> displayed_trees;
            if (ann_network.network.num_reticulations() == 0) {
                std::string newick = netrax::toExtendedNewick(ann_network);
                displayed_trees.emplace_back(std::make_pair(newick, 1.0));
            } else {
                for (int tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index) {
                    pll_utree_t* utree = netrax::displayed_tree_to_utree(ann_network.network, tree_index);
                    double prob = netrax::displayed_tree_prob(ann_network, tree_index);
                    Network displayedNetwork = netrax::convertUtreeToNetwork(*utree, 0);
                    std::string newick = netrax::toExtendedNewick(displayedNetwork);
                    pll_utree_destroy(utree, nullptr);
                    displayed_trees.emplace_back(std::make_pair(newick, prob));
                }
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

            if (ann_network.network.num_reticulations() > 0) {
                for (size_t i = 0; i < ann_network.reticulation_probs.size(); ++i) {
                    assert(ann_network.reticulation_probs[i] != 1.0);
                    assert(ann_network.reticulation_probs[i] != 0.0);
                }
            }

        } else {
            std::cout << "REMAINED BEST SCORE FOUND SO FAR: " << best_score << "\n";
        }

        if (new_score <= best_score) { // score did not get worse
            if (ann_network.network.num_reticulations() < ann_network.options.max_reticulations) {
                seen_improvement = true;
                NetraxInstance::add_extra_reticulations(ann_network, ann_network.network.num_reticulations() + 1);
                NetraxInstance::optimizeBranches(ann_network);
                NetraxInstance::optimizeModel(ann_network);
                NetraxInstance::updateReticulationProbs(ann_network);
                new_score = NetraxInstance::scoreNetwork(ann_network);
                std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network loglikelihood: " << NetraxInstance::computeLoglikelihood(ann_network) << "\n";
                std::cout << "Initial optimized " << ann_network.network.num_reticulations() << "-reticulation network BIC score: " << new_score << "\n";
            }
        }
    }

    std::cout << "The inferred network has " << best_num_reticulations << " reticulations and this BIC score: " << best_score << "\n\n";
    std::cout << "Best found network is:\n" << best_network << "\n\n";

    std::cout << "Statistics on which moves were taken:\n";
    for (const auto& entry : ann_network.stats.moves_taken) {
        std::cout << toString(entry.first) << ": " << entry.second << "\n";
    }
}

void run_single_start(NetraxOptions& netraxOptions, std::mt19937& rng) {
    double best_score = std::numeric_limits<double>::infinity();

    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);

    std::cout << "Initial network is:\n" << toExtendedNewick(ann_network) << "\n\n";
    std::string best_network = toExtendedNewick(ann_network);

    NetraxInstance::optimizeEverything(ann_network);
    double final_bic = NetraxInstance::scoreNetwork(ann_network);
    std::cout << "The inferred network has " << ann_network.network.num_reticulations() << " reticulations and this BIC score: " << final_bic << "\n\n";
    if (final_bic < best_score) {
        best_score = final_bic;
        std::cout << "IMPROVED BEST SCORE FOUND SO FAR: " << best_score << "\n\n";
        NetraxInstance::writeNetwork(ann_network, netraxOptions.output_file);
        best_network = toExtendedNewick(ann_network);
        std::cout << best_network << "\n";
        std::cout << "Better network written to " << netraxOptions.output_file << "\n";  
    } else {
        std::cout << "REMAINED BEST SCORE FOUND SO FAR: " << best_score << "\n";
    }

    std::cout << "Best found network is:\n" << best_network << "\n\n";
    std::cout << "Recomputing BIC on exactly this network gives: " << NetraxInstance::scoreNetwork(ann_network) << "\n";
}

void run_random(NetraxOptions& netraxOptions, std::mt19937& rng) {
    double best_score = std::numeric_limits<double>::infinity();
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t start_reticulations = 0;
    size_t n_iterations = 0;
    // random start networks
    while (true) {
        n_iterations++;
        std::cout << "Starting with new random network with " << start_reticulations << " reticulations.\n";
        netrax::AnnotatedNetwork ann_network = NetraxInstance::build_random_annotated_network(netraxOptions);
        NetraxInstance::init_annotated_network(ann_network, rng);
        NetraxInstance::add_extra_reticulations(ann_network, start_reticulations);
        NetraxInstance::optimizeEverythingInWaves(ann_network);
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
        } else if (n_iterations >= netraxOptions.num_random_start_networks) {
            break;
        }
    }

    // TODO: Get rid of the code duplication here
    // parsimony start networks
    n_iterations = 0;
    while (true) {
        n_iterations++;
        std::cout << "Starting with new parsimony tree with " << start_reticulations << " reticulations.\n";
        netrax::AnnotatedNetwork ann_network = NetraxInstance::build_parsimony_annotated_network(netraxOptions);
        NetraxInstance::init_annotated_network(ann_network, rng);
        NetraxInstance::add_extra_reticulations(ann_network, start_reticulations);
        NetraxInstance::optimizeEverythingInWaves(ann_network);
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
        } else if (n_iterations >= netraxOptions.num_parsimony_start_networks) {
            break;
        }
    }
}

void pretty_print(NetraxOptions& netraxOptions) {
    if (netraxOptions.start_network_file.empty()) {
        throw std::runtime_error("No input network specified to be pretty-printed");
    }
    Network network = netrax::readNetworkFromFile(netraxOptions.start_network_file);
    std::cout << exportDebugInfoNetwork(network) << "\n";
}

void score_only(NetraxOptions& netraxOptions, std::mt19937& rng) {
    if (netraxOptions.msa_file.empty()) {
        throw std::runtime_error("Need MSA to score a network");
    }
    if (netraxOptions.start_network_file.empty()) {
        throw std::runtime_error("Need network file to be scored");
    }
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);
    NetraxInstance::optimizeBranches(ann_network);
    NetraxInstance::updateReticulationProbs(ann_network);
    NetraxInstance::optimizeModel(ann_network);
    double final_bic = NetraxInstance::scoreNetwork(ann_network);
    double final_logl = NetraxInstance::computeLoglikelihood(ann_network);
    std::cout << "Number of reticulations: " << ann_network.network.num_reticulations() << "\n";
    std::cout << "BIC Score: " << final_bic << "\n";
    std::cout << "Loglikelihood: " << final_logl << "\n";
}

void extract_taxon_names(const NetraxOptions& netraxOptions) {
    if (netraxOptions.start_network_file.empty()) {
        throw std::runtime_error("Need network to extract taxon names");
    }
    netrax::Network network = netrax::readNetworkFromFile(netraxOptions.start_network_file,
            netraxOptions.max_reticulations);
    std::vector<std::string> tip_labels;
    for (size_t i = 0; i < network.num_tips(); ++i) {
        tip_labels.emplace_back(network.nodes_by_index[i]->getLabel());
    }
    std::cout << "Found " << tip_labels.size() << " taxa:\n";
    for (size_t i = 0; i < tip_labels.size(); ++i) {
        std::cout << tip_labels[i] << "\n";
    }
}

void extract_displayed_trees(NetraxOptions& netraxOptions, std::mt19937& rng) {
    if (netraxOptions.start_network_file.empty()) {
        throw std::runtime_error("Need network to extract displayed trees");
    }
    std::vector<std::pair<std::string, double> > displayed_trees;
    netrax::AnnotatedNetwork ann_network = NetraxInstance::build_annotated_network(netraxOptions);
    NetraxInstance::init_annotated_network(ann_network, rng);

    if (ann_network.network.num_reticulations() == 0) {
        std::string newick = netrax::toExtendedNewick(ann_network);
        displayed_trees.emplace_back(std::make_pair(newick, 1.0));
    } else {
        for (int tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index) {
            pll_utree_t* utree = netrax::displayed_tree_to_utree(ann_network.network, tree_index);
            double prob = netrax::displayed_tree_prob(ann_network, tree_index);
            Network displayedNetwork = netrax::convertUtreeToNetwork(*utree, 0);
            std::string newick = netrax::toExtendedNewick(displayedNetwork);
            pll_utree_destroy(utree, nullptr);
            displayed_trees.emplace_back(std::make_pair(newick, prob));
        }
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

void generate_random_network_only(NetraxOptions& netraxOptions, std::mt19937& rng) {
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
    //mpfr::mpreal::set_default_prec(mpfr::digits2bits(1000));

    //std::ios::sync_with_stdio(false);
    //std::cin.tie(NULL);
    netrax::NetraxOptions netraxOptions;

    //netrax::Network nw = netrax::readNetworkFromString("(((12:0.0381986,(((14:0.185353,(((((13:1.42035e-06)#0:1.43322e-06::0.5)#2:1.31229e-06::0.5)#3:1.4605e-06::0.5)#4:1.34252e-06::0.5)#5:0.0625::0.5):0.0926765)#1:0.0463383::0.5,#3:1::0.5):0.0463383):1.33647e-06,((((10:0.0445575,5:0.100001):1e-06,(3:0.140615,#4:1::0.5):0.140615):1e-06,#5:1::0.5):1e-06,#1:0.449939::0.5):1e-06):1e-06,11:0.0328376,(9:0.0343774,(#0:0.0935504::0.5,#2:1::0.5):0.0935504):1.33647e-06);");
    //std::cout << netrax::exportDebugInfo(nw) << "\n";

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

    if (netraxOptions.pretty_print_only) {
        pretty_print(netraxOptions);
        return 0;
    }

    if (netraxOptions.extract_taxon_names) {
        extract_taxon_names(netraxOptions);
        return 0;
    }

    if (netraxOptions.extract_displayed_trees) {
        extract_displayed_trees(netraxOptions, rng);
        return 0;
    }

    if (netraxOptions.generate_random_network_only) {
        generate_random_network_only(netraxOptions, rng);
        return 0;
    }

    if (netraxOptions.msa_file.empty()) {
        throw std::runtime_error("Need MSA to score a network");
    }

    if (netraxOptions.score_only) {
        score_only(netraxOptions, rng);
        return 0;
    } else if (netraxOptions.output_file.empty()) {
        throw std::runtime_error("No output path specified");
    }

    if (!netraxOptions.start_network_file.empty()) {
        run_single_start_waves(netraxOptions, rng);
    } else {
        run_random(netraxOptions, rng);
    }

    return 0;
}
