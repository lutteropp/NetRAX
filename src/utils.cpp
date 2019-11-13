/*
 * utils.cpp
 *
 *  Created on: Nov 13, 2019
 *      Author: Sarah Lutteropp
 */

#include "utils.hpp"

namespace netrax {

template <typename T>
void print_vector(const std::vector<T>& vector, const std::string& vectorName) {
	std::cout << vectorName << ":\n";
	for (size_t i = 0; i < vector.size(); ++i) {
		std::cout << vector[i];
		if (i < vector.size() - 1) {
			std::cout << ", ";
		}
	}
	std::cout << "\n";
}

void print_model_params(const pll_partition_t* partition) {
	for (size_t i = 0; i < partition->rate_matrices; ++i) {
		std::cout << "Parameters for rate matrix #" << i << ":\n";
		std::vector<double> basefreqs(partition->frequencies[i], partition->frequencies[i] + partition->states);
		print_vector<double>(basefreqs, "base freqs");
		size_t n_subst_rates = pllmod_util_subst_rate_count(partition->states);
		std::vector<double> subst_rates(partition->subst_params[i], partition->subst_params[i] + n_subst_rates);
		print_vector<double>(subst_rates, "subst rates");
	}
	if (partition->rate_cats > 1) {
		std::vector<double> ratecatRates(partition->rates, partition->rates + partition->rate_cats);
		print_vector<double>(ratecatRates, "ratecat rates");
		std::vector<double> ratecatWeights(partition->rate_weights, partition->rate_weights + partition->rate_cats);
		print_vector<double>(ratecatWeights, "ratecat weights");
	}
}

}
