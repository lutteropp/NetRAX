#pragma once

#include "../graph/AnnotatedNetwork.hpp"

namespace netrax {

size_t get_param_count(AnnotatedNetwork &ann_network);
size_t get_sample_size(AnnotatedNetwork &ann_network);
double aic(AnnotatedNetwork &ann_network, double logl);
double aicc(AnnotatedNetwork &ann_network, double logl);
double bic(AnnotatedNetwork &ann_network, double logl);
double scoreNetwork(AnnotatedNetwork &ann_network);
double scoreNetworkPseudo(AnnotatedNetwork &ann_network);

}  // namespace netrax