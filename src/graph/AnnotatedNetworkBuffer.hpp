#pragma once

#include <vector>
#include "AnnotatedNetwork.hpp"

namespace netrax {
    struct AnnotatedNetworkState {
        std::vector<std::vector<double> > brlens;
        std::vector<std::vector<double> > brprobs;
        std::vector<double> brlen_scalers;
        std::vector<std::vector<double> > pmatrix;
        std::vector<std::vector<double> > clv;
        BlobInformation blobInfo;
        std::vector<Node*> travbuffer;
    };

    AnnotatedNetworkState extractState(AnnotatedNetwork& ann_network);
    void applyState(AnnotatedNetwork& ann_network, AnnotatedNetworkState& state);
}