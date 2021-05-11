#include "ComplexityScoring.hpp"
#include "LikelihoodComputation.hpp"
#include "PseudoLoglikelihood.hpp"

namespace netrax {

double aic(double logl, double k) {
    return -2 * logl + 2 * k;
}
double aicc(double logl, double k, double n) {
    return aic(logl, k) + (2*k*k + 2*k) / (n - k - 1);
}
double bic(double logl, double k, double n) {
    return -2 * logl + k * log(n);
}

size_t get_param_count(AnnotatedNetwork& ann_network) {
    Network &network = ann_network.network;

    size_t param_count = ann_network.total_num_model_parameters;
    // reticulation probs as free parameters
    param_count += ann_network.network.num_reticulations();
    if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_UNLINKED) {
        assert(ann_network.fake_treeinfo->partition_count > 1);
        param_count += ann_network.fake_treeinfo->partition_count * network.num_branches();
    } else { // branch lengths are shared among partitions
        param_count += network.num_branches();
        if (ann_network.fake_treeinfo->brlen_linkage == PLLMOD_COMMON_BRLEN_SCALED) {
            assert(ann_network.fake_treeinfo->partition_count > 1);
            // each partition can scale the branch lengths by its own scaling factor
            param_count += ann_network.fake_treeinfo->partition_count - 1;
        }
    }
    return param_count;
}

size_t get_sample_size(AnnotatedNetwork& ann_network) {
    return ann_network.total_num_sites * ann_network.network.num_tips();
}

double aic(AnnotatedNetwork &ann_network, double logl) {
    return aic(logl, get_param_count(ann_network));
}

double aicc(AnnotatedNetwork &ann_network, double logl) {
    return aicc(logl, get_param_count(ann_network), get_sample_size(ann_network));
}

double bic(AnnotatedNetwork &ann_network, double logl) {
    return bic(logl, get_param_count(ann_network), get_sample_size(ann_network));
}

/**
 * Computes the BIC-score of a given network. A smaller score is a better score.
 * 
 * @param ann_network The network.
 */
double scoreNetwork(AnnotatedNetwork &ann_network) {
    double logl = computeLoglikelihood(ann_network, 1, 1);

    double bic_score = bic(ann_network, logl);
    if (bic_score == std::numeric_limits<double>::infinity()) {
        std::cout << "logl: " << logl << "\n";
        std::cout << "bic: " << bic_score << "\n";
        throw std::runtime_error("Invalid BIC score");
    }
    return bic_score;
}

double scoreNetworkPseudo(AnnotatedNetwork &ann_network) {
    double logl = computePseudoLoglikelihood(ann_network, 1, 1);

    double bic_score = bic(ann_network, logl);
    if (bic_score == std::numeric_limits<double>::infinity()) {
        std::cout << "logl: " << logl << "\n";
        std::cout << "bic: " << bic_score << "\n";
        throw std::runtime_error("Invalid BIC score");
    }
    return bic_score;
}

}