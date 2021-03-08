#include <cstddef>
#include <iostream>
#include <string>

#include "NetraxTest.hpp"

NetraxTest *env;

int main(int argc, char **argv) {
    env = new NetraxTest();
    //std::ios_base::sync_with_stdio(0);
    //std::cin.tie(0);

    // Set data dir using the program path.
    std::string call = argv[0];
    std::size_t found = call.find_last_of("/\\");
    if (found != std::string::npos) {
        env->data_dir = call.substr(0, found) + "/../data/";
    }

    ::testing::InitGoogleTest(&argc, argv);
//  MPI_INIT(&argc, &argv);
    ::testing::AddGlobalTestEnvironment(env);
    //::testing::GTEST_FLAG(filter) = "*BrlenOptTest.tree*";
    //:testing::GTEST_FLAG(filter) = "-NetworkIOTest.*";
    //::testing::GTEST_FLAG(filter) = "*SystemTest.allTree";
    //::testing::GTEST_FLAG(filter) = "*LikelihoodTest.celineNetwork:*LikelihoodTest.likelihoodFunctions*";
    //::testing::GTEST_FLAG(filter) = "*NetworkIOTest.reticulationHasLeafChild"*;
    //::testing::GTEST_FLAG(filter) = "*LikelihoodTest.smallNetworkWithRepeats";
    //::testing::GTEST_FLAG(filter) = "*LikelihoodTest.*";
    //::testing::GTEST_FLAG(filter) = "*MovesTest.deltaMinus*";
    ::testing::GTEST_FLAG(filter) = "*ValgrindTest.*";
    //::testing::GTEST_FLAG(filter) = "*SystemTest.allNetwork*";
    //::testing::GTEST_FLAG(filter) = "*SystemTest.random*";
    //::testing::GTEST_FLAG(filter) = "*SystemTest.problem*";

    // suppress any cout output while running the tests
    //std::cout.setstate(std::ios_base::failbit);
    auto result = RUN_ALL_TESTS();
    std::cout.clear();
//  MPI_FINALIZE();
    return result;
}
