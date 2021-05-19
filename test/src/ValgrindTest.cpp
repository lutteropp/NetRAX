#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/moves/Move.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/RaxmlWrapper.hpp"
#include "src/DebugPrintFunctions.hpp"

#include "src/helper/NetworkFunctions.hpp"

#include <gtest/gtest.h>
#include <string>
#include <mutex>
#include <iostream>

#include <raxml-ng/main.hpp>

#include "NetraxTest.hpp"

const std::string DATA_PATH = "/home/sarah/code-workspace/NetRAX/test/sample_networks/";

using namespace netrax;

void iterateOverClv(double* clv, ClvRangeInfo& clvInfo) {
    if (!clv) return;
    for (size_t i = 0; i < clvInfo.inner_clv_num_entries; ++i) {
        std::cout << clv[i] << " ";
    }
    std::cout << "\n";
}

void iterateOverScaler(unsigned int* scaler, ScaleBufferRangeInfo& scalerInfo) {
    if (!scaler) return;
    for (size_t i = 0; i < scalerInfo.scaler_size; ++i) {
        std::cout << scaler[i] << " ";
    }
    std::cout << "\n";
}

TEST (ValgrindTest, valgrind) {
    ClvRangeInfo clvRangeInfo;
    clvRangeInfo.alignment = 16;
    clvRangeInfo.start = 0;
    clvRangeInfo.end = 19;
    clvRangeInfo.total_num_clvs = 20;
    clvRangeInfo.inner_clv_num_entries = 100;
    double* clv_vector = create_single_empty_clv(clvRangeInfo);
    iterateOverClv(clv_vector, clvRangeInfo);
    pll_aligned_free(clv_vector);
}