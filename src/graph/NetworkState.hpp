#pragma once

#include <vector>

struct NetworkState {
    std::vector<std::vector<double> > brlens;
    std::vector<std::vector<double> > brprobs;
    std::vector<std::vector<double> > pmatrix;
    std::vector<std::vector<double> > clv_vectors;
};