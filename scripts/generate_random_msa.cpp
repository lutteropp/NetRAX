#include <CLI11.hpp>
#include <fstream>
#include <random>
#include <string>
#include "../src/io/NetworkIO.hpp"
#include "../src/graph/Network.hpp"

struct RandomMSAOptions {
    std::string newickPath;
    std::string msaOutPath;
    size_t msa_width;
    double gappyness; // in [0,1]
};

int parseOptions(int argc, char **argv, RandomMSAOptions *options) {
    CLI::App app { "NetRAX: Phylogenetic Network Inference without Incomplete Lineage Sorting" };
    app.add_option("--newick", options->newickPath, "The newick input file")->required();
    app.add_option("--outfile", options->msaOutPath, "The output file for the generated MSA");
    app.add_option("-w,--width", options->msa_width, "The width of the MSA")->required();
    app.add_option("-g,--gappyness", options->gappyness, "The percentage of gaps, double value between 0 and 1")->required();
    CLI11_PARSE(app, argc, argv);
    assert(options->gappyness >= 0 && options->gappyness <= 1.0);
    return 0;
}

std::string genRandomString(const RandomMSAOptions &options) {
    std::string res;
    std::default_random_engine gen((std::random_device())());
    for (size_t i = 0; i < options.msa_width; ++i) {
        double gap = std::uniform_real_distribution<double> { 0, 1 }(gen);
        if (gap < options.gappyness) {
            res += '-';
        } else {
            int baseIdx = std::uniform_int_distribution<int> { 0, 3 }(gen);
            switch (baseIdx) {
            case 0:
                res += 'A';
                break;
            case 1:
                res += 'C';
                break;
            case 2:
                res += 'G';
                break;
            case 3:
                res += 'T';
                break;
            default:
                throw std::runtime_error("shouldn't be here");
            }
        }
    }
    return res;
}

int main(int argc, char **argv) {
    RandomMSAOptions options;
    parseOptions(argc, argv, &options);
    netrax::Network network = netrax::readNetworkFromFile(options.newickPath);
    std::ofstream outfile(options.msaOutPath);
    std::cout << "Number of network tip nodes: " << network.num_tips() << "\n";
    for (size_t i = 0; i < network.num_tips(); ++i) {
        const netrax::Node &node = network.nodes[i];
        outfile << ">" << node.label << "\n";
        std::string seq = genRandomString(options);
        outfile << seq << "\n";

        std::cout << ">" << node.label << "\n";
        std::cout << seq << "\n";
    }
    outfile.close();
    return 0;
}
