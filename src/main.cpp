#include <iostream>
#include <string>
#include <random>
#include <limits>
#include <functional>

#include <CLI11.hpp>
#include <raxml-ng/main.hpp>

#include "optimization/Optimization.hpp"

#include "likelihood/mpreal.h"
#include "graph/AnnotatedNetwork.hpp"
#include "NetraxOptions.hpp"
#include "io/NetworkIO.hpp"
#include "DebugPrintFunctions.hpp"
#include "likelihood/LikelihoodComputation.hpp"
#include "optimization/ModelOptimization.hpp"
#include "search/NetworkSearch.hpp"
#include "NetworkDistances.hpp"
#include "helper/NetworkFunctions.hpp"

#include "../RaxmlWrapper.hpp"

#include "likelihood/ComplexityScoring.hpp"

#ifdef WITHGPERFTOOLS
#include <gperftools/profiler.h>
#endif

using namespace netrax;

void error_exit(std::string msg) {
    mpfr_free_cache();
    ParallelContext::finalize(true);
    throw std::runtime_error(msg);
}

int parseOptions(int argc, char **argv, netrax::NetraxOptions *options)
{
    CLI::App app{"NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting"};
    app.add_option("--msa", options->msa_file, "The Multiple Sequence Alignment File");
    app.add_option("--model", options->model_file, "The partitions assignment in case of a partitioned MSA.");
    app.add_option("-o,--output", options->output_file, "File where to write the final network to");
    app.add_option("--start_network", options->start_network_file, "A network file (in Extended Newick format) to start the search on");
    app.add_option("-r,--max_reticulations", options->max_reticulations,
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
    app.add_flag("--check_weird_network", options->check_weird_network, "Only check if the network has displayed trees with equal bipartitions");
    app.add_flag("--extract_taxon_names", options->extract_taxon_names, "Only extract all taxon names from a network.");
    app.add_flag("--generate_random_network_only", options->generate_random_network_only, "Only generate a random network, with as many reticulations as specified in the -r parameter");
    app.add_flag("--pretty_print_only", options->pretty_print_only, "Only pretty-print a given input network.");
    app.add_option("--scale_branches_only", options->scale_branches_only, "Only scale branches of a given network by a given factor.");

    app.add_flag("--network_distance_only", options->network_distance_only, "Only compute unrooted softwired network distance.");
    app.add_option("--first_network", options->first_network_path, "Path to first network file for distance computation.");
    app.add_option("--second_network", options->second_network_path, "Path to second network file for distance computation.");

    app.add_flag("--change_reticulation_prob_only", options->change_reticulation_probs_only, "Only change the reticulation probs of the input network.");
    app.add_option("--overwritten_reticulation_prob", options->overwritten_reticulation_prob, "New probability to use for the reticulations in overwrite-only mode.");

    std::string brlen_linkage = "linked";
    app.add_option("--brlen", brlen_linkage, "branch length linkage between partitions (linked, scaled, or unlinked) (default: linked)");

    bool average_displayed_tree_variant = false;
    bool best_displayed_tree_variant = false;
    app.add_flag("--average_displayed_tree_variant", average_displayed_tree_variant, "Use weighted average instead of only best displayed tree in network likelihood formula.");
    app.add_flag("--best_displayed_tree_variant", best_displayed_tree_variant, "Use best displayed tree instead of weighted average in network likelihood formula.");
    app.add_flag("--no_prefiltering", options->no_prefiltering, "Disable prefiltering of highly-promising move candidates.");
    app.add_option("--prefilter_keep", options->prefilter_keep, "Number of promising candidates to keep after prefiltering (default: 60)");
    app.add_option("--rank_keep", options->prefilter_keep, "Number of promising candidates to keep after ranking (default: 20)");

    app.add_option("--greedy_factor", options->greedy_factor, "Instantly accept a move if it improves BIC by more than the given factor (default: infinity). Gives (maybe faster) results with (maybe worse) inference quality. Needs to be greater-equal than 1.");
    app.add_flag("--reorder_candidates", options->reorder_candidates, "Reorder move candidates by proximity to last accepted move.");
    
    app.add_flag("--old_wavesearch", options->old_wavesearch, "Use the old wavesearch algorithm.");

    app.add_flag("--slow_mode", options->slow_mode, "Enable slow mode.");
    app.add_flag("--save_memory", options->save_memory, "Save some memory.");
    app.add_flag("--no_elbow_method", options->no_elbow_method, "No elbow method.");

    app.add_option("--max_better_candidates", options->max_better_candidates, "Stop candidate evaluation early if this many candidates with better BIC have been found already.");

    app.add_flag("--no_rnni_moves", options->no_rnni_moves, "Do not use rNNI moves.");
    app.add_flag("--no_rspr_moves", options->no_rspr_moves, "Do not use rSPR moves.");
    app.add_flag("--no_arc_insertion_moves", options->no_arc_insertion_moves, "Use only DeltaPlus moves instead of full ArcInsertion moves (faster, but worse inference quality).");

    app.add_flag("--enforce_extra_search", options->enforce_extra_search, "After finishing the normal search, keep searching by enforcing an extra reticulation.");

    app.add_option("--scrambling", options->scrambling, "Maximum failed consecutive scrambling retries for escaping out of local maxima (default: 0).");
    app.add_option("--scrambling_radius", options->scrambling_radius, "Number of random rSPR moves to apply when scrambling a network (default: 2).");

    app.add_flag("--sim_anneal", options->sim_anneal, "Use simulated annealing instead of hill climbing during network topology search.");
    app.add_option("--start_temperature", options->start_temperature, "Start temperature to be used for simulated annealing (default: 100).");

    //app.add_flag("--extreme_greedy_prefiltering", options->extreme_greedy_prefiltering, "Enable extreme greedy prefiltering mode.");

    CLI11_PARSE(app, argc, argv);

    if (average_displayed_tree_variant && best_displayed_tree_variant) {
        error_exit("Cannot specify both --average_displayed_tree_variant and --best_displayed_tree_variant at once");
    }
    options->likelihood_variant = (average_displayed_tree_variant) ? LikelihoodVariant::AVERAGE_DISPLAYED_TREES : LikelihoodVariant::BEST_DISPLAYED_TREE;
    //options->computePseudo = (options->likelihood_variant == LikelihoodVariant::SARAH_PSEUDO);

    if (brlen_linkage == "scaled")
    {
        options->brlen_linkage = PLLMOD_COMMON_BRLEN_SCALED;
    }
    else if (brlen_linkage == "linked")
    {
        options->brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
    }
    else if (brlen_linkage == "unlinked")
    {
        options->brlen_linkage = PLLMOD_COMMON_BRLEN_UNLINKED;
    }
    else
    {
        error_exit("brlen_linkage needs to be one of {linked, scaled, unlinked}");
    }

    if (options->greedy_factor < 1.0) {
        error_exit("greedy_factor needs to be at least 1.0");
    }

    if (options->prefilter_keep < 1) {
        error_exit("prefilter keep must be at least 1");
    }

    assert(!options->use_repeats);
    return 0;
}

std::vector<MoveType> getTypesBySpeedGoodStart(const NetraxOptions& options) {
    //std::vector<MoveType> typesBySpeed = {MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::HeadMove, MoveType::TailMove, MoveType::ArcRemovalMove, MoveType::DeltaPlusMove, MoveType::ArcInsertionMove};
    std::vector<MoveType> typesBySpeed;
    if (!options.no_arc_removal_moves) {
        typesBySpeed.emplace_back(MoveType::ArcRemovalMove);
    }
    if (!options.no_arc_insertion_moves) {
        typesBySpeed.emplace_back(MoveType::ArcInsertionMove);
    } else {
        typesBySpeed.emplace_back(MoveType::DeltaPlusMove);
    }
    if (!options.no_rspr_moves) {
        typesBySpeed.emplace_back(MoveType::RSPRMove);
    }
    if (!options.no_rnni_moves) {
        typesBySpeed.emplace_back(MoveType::RNNIMove);
    }
    if (!options.no_arc_removal_moves) {
        typesBySpeed.emplace_back(MoveType::ArcRemovalMove);
    }
    return typesBySpeed;
}

std::vector<MoveType> getTypesBySpeed(const NetraxOptions& options) {
    //std::vector<MoveType> typesBySpeed = {MoveType::RNNIMove, MoveType::RSPR1Move, MoveType::HeadMove, MoveType::TailMove, MoveType::ArcRemovalMove, MoveType::DeltaPlusMove, MoveType::ArcInsertionMove};
    std::vector<MoveType> typesBySpeed;
    if (!options.no_arc_removal_moves) {
        typesBySpeed.emplace_back(MoveType::ArcRemovalMove);
    }
    if (!options.no_rspr_moves) {
        typesBySpeed.emplace_back(MoveType::RSPRMove);
    }
    if (!options.no_rnni_moves) {
        typesBySpeed.emplace_back(MoveType::RNNIMove);
    }
    if (!options.no_arc_insertion_moves) {
        typesBySpeed.emplace_back(MoveType::ArcInsertionMove);
    } else {
        typesBySpeed.emplace_back(MoveType::DeltaPlusMove);
    }
    if (!options.no_rspr_moves) {
        typesBySpeed.emplace_back(MoveType::RSPRMove);
    }
    if (!options.no_rnni_moves) {
        typesBySpeed.emplace_back(MoveType::RNNIMove);
    }
    if (!options.no_arc_removal_moves) {
        typesBySpeed.emplace_back(MoveType::ArcRemovalMove);
    }
    return typesBySpeed;
}

void pretty_print(const NetraxOptions &netraxOptions)
{
    Network network = netrax::readNetworkFromFile(netraxOptions.start_network_file, netraxOptions);
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << exportDebugInfoNetwork(network) << "\n";
    }
}

void score_only(NetraxOptions &netraxOptions, const RaxmlInstance& instance, std::mt19937 &rng)
{
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions, instance);
    init_annotated_network(ann_network, rng);
    optimizeModel(ann_network);

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Initial, given network:\n";
        std::cout << toExtendedNewick(ann_network) << "\n";
    }

    double start_bic = scoreNetwork(ann_network);
    double start_logl = computeLoglikelihood(ann_network, 1, 1);
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Initial (before brlen and reticulation opt) BIC Score: " << start_bic << "\n";
        std::cout << "Initial (before brlen and reticulation opt) loglikelihood: " << start_logl << "\n";
    }

    optimizeAllNonTopology(ann_network, true);

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Network after optimization of brlens and reticulation probs:\n";
        std::cout << toExtendedNewick(ann_network) << "\n";
    }

    double final_bic = scoreNetwork(ann_network);
    double final_logl = computeLoglikelihood(ann_network, 1, 1);
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Number of reticulations: " << ann_network.network.num_reticulations() << "\n";
        std::cout << "BIC Score: " << final_bic << "\n";
        std::cout << "Loglikelihood: " << final_logl << "\n";
    }
}

void extract_taxon_names(const NetraxOptions &netraxOptions)
{
    netrax::Network network = netrax::readNetworkFromFile(netraxOptions.start_network_file, netraxOptions,
                                                          netraxOptions.max_reticulations);
    std::vector<std::string> tip_labels;
    for (size_t i = 0; i < network.num_tips(); ++i)
    {
        tip_labels.emplace_back(network.nodes_by_index[i]->getLabel());
    }
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Found " << tip_labels.size() << " taxa:\n";
        for (size_t i = 0; i < tip_labels.size(); ++i)
        {
            std::cout << tip_labels[i] << "\n";
        }
    }
}

void extract_displayed_trees(NetraxOptions &netraxOptions, const RaxmlInstance& instance, std::mt19937 &rng)
{
    std::vector<std::pair<std::string, double>> displayed_trees;
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions, instance);
    init_annotated_network(ann_network, rng);

    if (ann_network.network.num_reticulations() == 0)
    {
        std::string newick = netrax::toExtendedNewick(ann_network);
        displayed_trees.emplace_back(std::make_pair(newick, 1.0));
    }
    else
    {
        for (int tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index)
        {
            pll_utree_t *utree = netrax::displayed_tree_to_utree(ann_network.network, tree_index);
            double prob = netrax::displayed_tree_prob(ann_network, tree_index);
            Network displayedNetwork = netrax::convertUtreeToNetwork(*utree, netraxOptions, 0);
            std::string newick = netrax::toExtendedNewick(displayedNetwork);
            pll_utree_destroy(utree, nullptr);
            displayed_trees.emplace_back(std::make_pair(newick, prob));
        }
    }
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Number of displayed trees: " << displayed_trees.size() << "\n";
        std::cout << "Displayed trees Newick strings:\n";
        for (const auto &entry : displayed_trees)
        {
            std::cout << entry.first << "\n";
        }
        std::cout << "Displayed trees probabilities:\n";
        for (const auto &entry : displayed_trees)
        {
            std::cout << entry.second << "\n";
        }
    }
}

void scale_branches_only(NetraxOptions &netraxOptions, const RaxmlInstance& instance, std::mt19937 &rng)
{
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions, instance);
    init_annotated_network(ann_network, rng);
    ann_network.fake_treeinfo->brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;

    for (size_t i = 0; i < ann_network.network.num_branches(); ++i)
    {
        ann_network.network.edges_by_index[i]->length *= netraxOptions.scale_branches_only;
        ann_network.fake_treeinfo->linked_branch_lengths[i] *= netraxOptions.scale_branches_only;
    }
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        writeNetwork(ann_network, netraxOptions.output_file);
        std::cout << "Network with scaled branch lengths written to " << netraxOptions.output_file << "\n";
    }
}

void network_distance_only(NetraxOptions &netraxOptions, const RaxmlInstance& instance, std::mt19937 &rng)
{
    netrax::AnnotatedNetwork ann_network_1 = build_annotated_network_from_file(netraxOptions, instance, netraxOptions.first_network_path);
    init_annotated_network(ann_network_1, rng);
    netrax::AnnotatedNetwork ann_network_2 = build_annotated_network_from_file(netraxOptions, instance, netraxOptions.second_network_path);
    init_annotated_network(ann_network_2, rng);
    if (ann_network_1.network.num_tips() != ann_network_2.network.num_tips())
    {
        throw std::runtime_error("Unequal number of taxa");
    }
    std::unordered_map<std::string, unsigned int> label_to_int;
    for (size_t i = 0; i < ann_network_1.network.num_tips(); ++i)
    {
        label_to_int[ann_network_1.network.nodes_by_index[i]->label] = i;
    }

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Unrooted softwired network distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::UNROOTED_SOFTWIRED_DISTANCE) << "\n";
        std::cout << "Unrooted hardwired network distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::UNROOTED_HARDWIRED_DISTANCE) << "\n";
        std::cout << "Unrooted displayed trees distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::UNROOTED_DISPLAYED_TREES_DISTANCE) << "\n";

        std::cout << "Rooted softwired network distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::ROOTED_SOFTWIRED_DISTANCE) << "\n";
        std::cout << "Rooted hardwired network distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::ROOTED_HARDWIRED_DISTANCE) << "\n";
        std::cout << "Rooted displayed trees distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::ROOTED_DISPLAYED_TREES_DISTANCE) << "\n";
        std::cout << "Rooted tripartition distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::ROOTED_TRIPARTITION_DISTANCE) << "\n";
        std::cout << "Rooted path multiplicity distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::ROOTED_PATH_MULTIPLICITY_DISTANCE) << "\n";
        std::cout << "Rooted nested labels distance: " << get_network_distance(ann_network_1, ann_network_2, label_to_int, NetworkDistanceType::ROOTED_NESTED_LABELS_DISTANCE) << "\n";
    }
}

void check_weird_network(NetraxOptions &netraxOptions, const RaxmlInstance& instance, std::mt19937 &rng)
{
    std::vector<pll_utree_t *> displayed_trees;
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions, instance);
    init_annotated_network(ann_network, rng);

    for (int tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index)
    {
        pll_utree_t *utree = netrax::displayed_tree_to_utree(ann_network.network, tree_index);
        displayed_trees.emplace_back(utree);
    }

    unsigned int n_tips = ann_network.network.num_tips();
    unsigned int n_pairs = 0;
    unsigned int n_equal = 0;
    for (size_t i = 0; i < displayed_trees.size(); ++i)
    {
        for (size_t j = i + 1; j < displayed_trees.size(); ++j)
        {
            n_pairs++;

            unsigned int rf_dist = pllmod_utree_rf_distance(displayed_trees[i]->vroot, displayed_trees[j]->vroot, n_tips);
            if (rf_dist == 0)
            {
                n_equal++;
            }
        }
    }
    for (size_t i = 0; i < displayed_trees.size(); ++i)
    {
        pll_utree_destroy(displayed_trees[i], nullptr);
    }

    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        std::cout << "Number of pairs: " << n_pairs << "\n";
        std::cout << "Number of equal pairs: " << n_equal << "\n";
    }
}

void generate_random_network_only(NetraxOptions &netraxOptions, const RaxmlInstance& instance, std::mt19937 &rng)
{
    std::uniform_int_distribution<long> dist(0, RAND_MAX);
    netrax::AnnotatedNetwork ann_network = build_random_annotated_network(netraxOptions, instance, dist(rng));
    init_annotated_network(ann_network, rng);
    ann_network.fake_treeinfo->brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
    add_extra_reticulations(ann_network, netraxOptions.max_reticulations);
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        writeNetwork(ann_network, netraxOptions.output_file);
        std::cout << "Final network written to " << netraxOptions.output_file << "\n";
    }
}

void scale_reticulation_probs_only(NetraxOptions &netraxOptions, const RaxmlInstance& instance, std::mt19937 &rng)
{
    netrax::AnnotatedNetwork ann_network = build_annotated_network(netraxOptions, instance);
    init_annotated_network(ann_network, rng);
    ann_network.fake_treeinfo->brlen_linkage = PLLMOD_COMMON_BRLEN_LINKED;
    for (size_t i = 0; i < ann_network.network.num_reticulations(); ++i) {
        ann_network.reticulation_probs[i] = netraxOptions.overwritten_reticulation_prob;
    }
    if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
        writeNetwork(ann_network, netraxOptions.output_file);
        std::cout << "Final network written to " << netraxOptions.output_file << "\n";
    }
}

bool quick_function(NetraxOptions& netraxOptions, const RaxmlInstance& instance) {
    std::mt19937 rng(netraxOptions.seed);
    if (netraxOptions.pretty_print_only) {
        if (netraxOptions.start_network_file.empty())
        {
            error_exit("No input network specified to be pretty-printed");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            pretty_print(netraxOptions);
        }
        return true;
    } else if (netraxOptions.extract_displayed_trees) {
        if (netraxOptions.start_network_file.empty())
        {
            error_exit("Need network to extract displayed trees");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            extract_displayed_trees(netraxOptions, instance, rng);
        }
        return true;
    } else if (netraxOptions.check_weird_network) {
        if (netraxOptions.start_network_file.empty())
        {
            error_exit("Need network to extract displayed trees");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            check_weird_network(netraxOptions, instance, rng);
        }
        return true;
    } else if (netraxOptions.generate_random_network_only) {
        if (netraxOptions.msa_file.empty())
        {
            error_exit("Need MSA to decide on the number of taxa");
        }
        if (netraxOptions.output_file.empty())
        {
            error_exit("Need output file to write the generated network");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            generate_random_network_only(netraxOptions, instance, rng);
        }
        return true;
    } else if (netraxOptions.scale_branches_only != 0.0) {
        if (netraxOptions.start_network_file.empty())
        {
            error_exit("Need network to scale branches");
        }
        if (netraxOptions.output_file.empty())
        {
            error_exit("Need output file to write the scaled network");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            scale_branches_only(netraxOptions, instance, rng);
        }
        return true;
    } else if (netraxOptions.change_reticulation_probs_only) {
        if (netraxOptions.msa_file.empty())
        {
            error_exit("Need MSA to decide on the number of taxa");
        }
        if (netraxOptions.output_file.empty())
        {
            error_exit("Need output file to write the generated network");
        }
        if (netraxOptions.start_network_file.empty()) {
            error_exit("Need start network file");
        }
        if (netraxOptions.overwritten_reticulation_prob < 0.0 || netraxOptions.overwritten_reticulation_prob > 1.0) {
            error_exit("new prob has to be in [0,1]");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            scale_reticulation_probs_only(netraxOptions, instance, rng);
        }
        return true;
    } else if (netraxOptions.network_distance_only) {
        if (netraxOptions.first_network_path.empty() || netraxOptions.second_network_path.empty())
        {
            error_exit("Need networks to compute distance");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            network_distance_only(netraxOptions, instance, rng);
        }
        return true;
    } 
    return false;
}

void netrax_thread_main(NetraxOptions& netraxOptions, const RaxmlInstance& instance) {
    /* wait until master thread prepares all global data */
    //printf("WORKER: %u, LOCAL_THREAD: %u\n", ParallelContext::group_id(), ParallelContext::local_proc_id());
    ParallelContext::global_barrier();

    if (no_parallelization_needed(netraxOptions)) {
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) {
            quick_function(netraxOptions, instance);
        }
        return;
    }

    //check_oversubscribe(instance);

    std::mt19937 rng(netraxOptions.seed);
    srand(netraxOptions.seed);

    if (netraxOptions.score_only)
    {
        score_only(netraxOptions, instance, rng);
    }
    else if (!netraxOptions.start_network_file.empty())
    {
        std::vector<MoveType> typesBySpeed = getTypesBySpeed(netraxOptions);
        std::vector<MoveType> typesBySpeedGoodStart = getTypesBySpeedGoodStart(netraxOptions);
        run_single_start_waves(netraxOptions, instance, typesBySpeed, typesBySpeedGoodStart, rng);
    } else
    {
        std::vector<MoveType> typesBySpeed = getTypesBySpeed(netraxOptions);
        run_random(netraxOptions, instance, typesBySpeed, rng);
    }
    ParallelContext::global_barrier();
}

void setup_parallel_stuff(NetraxOptions& netraxOptions, RaxmlInstance& instance) {
    if (no_parallelization_needed(netraxOptions)) {
        return;
    }
    autotune_threads(instance);
    check_options(instance);

    auto thread_function = std::bind(netrax_thread_main,
                                                 std::ref(netraxOptions),
                                                 std::ref(instance));
    ParallelContext::init_pthreads(instance.opts, thread_function);

    /*std::cout << "num_threads: " << ParallelContext::num_threads() << "\n";
    std::cout << "num workers: " << instance.opts.num_workers << "\n";
    std::cout << "num local groups: " << ParallelContext::num_local_groups() << "\n";
    std::cout << "num ranks: " << ParallelContext::num_ranks() << "\n";
    std::cout << "num groups: " << ParallelContext::num_groups() << "\n";
    std::cout << "threads per group: " << ParallelContext::threads_per_group() << "\n";
    std::cout << "num procs: " << ParallelContext::num_procs() << "\n";*/

    /* init workers */
    assert(instance.opts.num_workers > 0);
    for (size_t i = 0; i < ParallelContext::num_local_groups(); ++i)
    {
        const auto& grp = ParallelContext::thread_group(i);
        instance.workers.emplace_back(instance, grp.group_id);
    }

    init_parallel_buffers(instance);

    balance_load(instance);

    //std::cout << instance.proc_part_assign << "\n";

    /* lazy-load part of the alignment assigned to the current MPI rank */
    if (instance.opts.msa_format == FileFormat::binary && instance.opts.use_rba_partload)
    {
        // doesn't work with coarse-grained parallelization!
        assert(ParallelContext::num_groups() == 1);

        // collect PartitionAssignments from all worker threads
        PartitionAssignment local_part_ranges;
        for (size_t i = 0; i < instance.opts.num_threads; ++i)
        {
        auto thread_ranges = instance.proc_part_assign.at(ParallelContext::local_proc_id() + i);
        for (auto& r: thread_ranges)
        {
            local_part_ranges.assign_sites(r.part_id, r.start, r.length);
        }
        }

        LOG_DEBUG << "Loading MSA segments from RBA file..." << endl;
        RBAStream bs(instance.opts.msa_file);
        auto& parted_msa = *instance.parted_msa;
        bs >> RBAStream::RBAOutput(parted_msa, RBAStream::RBAElement::seqdata, &local_part_ranges);
    }

    //balance_load_coarse(instance, cm.checkp_file());
}

int internal_main_netrax(int argc, char **argv, void* comm)
{
    ParallelContext::init_mpi(argc, argv, comm);

    std::cout << std::setprecision(10);
    //mpfr::mpreal::set_default_prec(mpfr::digits2bits(1000));

    //std::ios::sync_with_stdio(false);
    //std::cin.tie(NULL);
    netrax::NetraxOptions netraxOptions;

    netraxOptions.num_ranks = ParallelContext::num_ranks();

    //netrax::Network nw = netrax::readNetworkFromString("(((12:0.0381986,(((14:0.185353,(((((13:1.42035e-06)#0:1.43322e-06::0.5)#2:1.31229e-06::0.5)#3:1.4605e-06::0.5)#4:1.34252e-06::0.5)#5:0.0625::0.5):0.0926765)#1:0.0463383::0.5,#3:1::0.5):0.0463383):1.33647e-06,((((10:0.0445575,5:0.100001):1e-06,(3:0.140615,#4:1::0.5):0.140615):1e-06,#5:1::0.5):1e-06,#1:0.449939::0.5):1e-06):1e-06,11:0.0328376,(9:0.0343774,(#0:0.0935504::0.5,#2:1::0.5):0.0935504):1.33647e-06);");
    //std::cout << netrax::exportDebugInfo(nw) << "\n";

    if (argc == 1) {
        char * temp[] = {argv[0],strdup("-h")};
        parseOptions(2, temp, &netraxOptions);
        free(temp[1]);
        mpfr_free_cache();
        ParallelContext::finalize();
        return 0;
    }
    parseOptions(argc, argv, &netraxOptions);

    // early stop if the user only wanted list of commands
    for(int i=0;i<argc;i++)
    {
        if(string(argv[i]) == "-h" || string(argv[i]) == "--help")
        {
            mpfr_free_cache();
            ParallelContext::finalize();
            return 0;
        }
    }

    if (netraxOptions.extract_taxon_names) {
        if (netraxOptions.start_network_file.empty())
        {
            error_exit("Need network to extract taxon names");
        }
        if (ParallelContext::master_rank() && ParallelContext::master_thread()) { // only the master rank does the simple work
            extract_taxon_names(netraxOptions);
        }
        mpfr_free_cache();
        ParallelContext::finalize();
        return 0;
    }

    RaxmlInstance instance = createRaxmlInstance(netraxOptions);
    
    // make sure all MPI ranks use the same random seed
    //ParallelContext::mpi_broadcast(&netraxOptions.seed, sizeof(long));

    if (!no_parallelization_needed(netraxOptions)) {
        if (netraxOptions.msa_file.empty())
        {
            error_exit("Need MSA to score a network");
        }
        if (netraxOptions.score_only) {
            if (netraxOptions.msa_file.empty())
            {
                error_exit("Need MSA to score a network");
            }
            if (netraxOptions.start_network_file.empty())
            {
                error_exit("Need network file to be scored");
            }
        } else if (netraxOptions.output_file.empty()) {
            error_exit("No output path specified");
        }
    }

    //logger().add_log_stream(&std::cout);

    setup_parallel_stuff(netraxOptions, instance);
    netrax_thread_main(netraxOptions, instance);

    mpfr_free_cache();
    ParallelContext::finalize();

    return 0;
}

#ifdef _NETRAX_BUILD_AS_LIB

extern "C" int dll_main(int argc, char** argv, void* comm)
{
  return internal_main_netrax(argc, argv, comm);
}

#else

int main(int argc, char** argv)
{
#ifdef WITHGPERFTOOLS
    ProfilerStart("profile.log");
#endif
  auto retval = internal_main_netrax(argc, argv, 0);
#ifdef WITHGPERFTOOLS
    ProfilerStop();
#endif
  return retval;
}

#endif