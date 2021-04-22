/*
 * utils.cpp
 *
 *  Created on: Nov 13, 2019
 *      Author: Sarah Lutteropp
 */

#include "utils.hpp"

#include <limits>

namespace netrax {

bool approximatelyEqual(double a, double b)
{
    double epsilon = std::numeric_limits<double>::epsilon();
    return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool essentiallyEqual(double a, double b)
{
    double epsilon = std::numeric_limits<double>::epsilon();
    return fabs(a - b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool definitelyGreaterThan(double a, double b)
{
    double epsilon = std::numeric_limits<double>::epsilon();
    return (a - b) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool definitelyLessThan(double a, double b)
{
    double epsilon = std::numeric_limits<double>::epsilon();
    return (b - a) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<typename T>
void print_vector(const std::vector<T> &vector, const std::string &vectorName) {
    std::cout << vectorName << ":\n";
    for (size_t i = 0; i < vector.size(); ++i) {
        std::cout << vector[i];
        if (i < vector.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";
}

void print_model_params(const pll_partition_t &partition) {
    for (size_t i = 0; i < partition.rate_matrices; ++i) {
        std::cout << "Parameters for rate matrix #" << i << ":\n";
        std::vector<double> basefreqs(partition.frequencies[i],
                partition.frequencies[i] + partition.states);
        print_vector<double>(basefreqs, "base freqs");
        size_t n_subst_rates = pllmod_util_subst_rate_count(partition.states);
        std::vector<double> subst_rates(partition.subst_params[i],
                partition.subst_params[i] + n_subst_rates);
        print_vector<double>(subst_rates, "subst rates");
    }
    if (partition.rate_cats > 1) {
        std::vector<double> ratecatRates(partition.rates, partition.rates + partition.rate_cats);
        print_vector<double>(ratecatRates, "ratecat rates");
        std::vector<double> ratecatWeights(partition.rate_weights,
                partition.rate_weights + partition.rate_cats);
        print_vector<double>(ratecatWeights, "ratecat weights");
    }
}

void print_model_params(const pllmod_treeinfo_t &treeinfo) {
    for (size_t i = 0; i < treeinfo.partition_count; ++i) {
        std::cout << "model params for partition #" << i << ":\n";
        print_model_params(*(treeinfo.partitions[i]));
        std::cout << "alpha: " << treeinfo.alphas[i] << "\n";
    }
}

template<typename T>
void assign_c_vector(const T *from, T *to, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        to[i] = from[i];
    }
}

void transfer_model_params(const pll_partition_t &from, pll_partition_t *to) {
    for (size_t i = 0; i < from.rate_matrices; ++i) {
        assert(from.states == to->states);
        assign_c_vector<double>(from.frequencies[i], to->frequencies[i], from.states);
        size_t n_subst_rates = pllmod_util_subst_rate_count(from.states);
        assert(n_subst_rates == pllmod_util_subst_rate_count(to->states));
        assign_c_vector<double>(from.subst_params[i], to->subst_params[i], n_subst_rates);
    }
    if (from.rate_cats > 1) {
        assert(from.rate_cats == to->rate_cats);
        assign_c_vector<double>(from.rates, to->rates, from.rate_cats);
        assign_c_vector<double>(from.rate_weights, to->rate_weights, from.rate_cats);
    }
    to->prop_invar[0] = from.prop_invar[0];
}

void transfer_model_params(const pllmod_treeinfo_t &from, pllmod_treeinfo_t *to) {
    assert(to);
    assert(from.partition_count == to->partition_count);
    size_t pcnt = from.partition_count;
    for (size_t i = 0; i < pcnt; ++i) {
        transfer_model_params(*(from.partitions[i]), to->partitions[i]);
        to->alphas[i] = from.alphas[i];
        if (from.brlen_scalers) {
            to->brlen_scalers[i] = from.brlen_scalers[i];
        }
    }
}
}
