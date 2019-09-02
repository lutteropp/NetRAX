#include <gtest/gtest.h>
#include <string>
#include <iostream>

#include "NetraxTest.hpp"

NetraxTest* env;

int main(int argc, char** argv)
{
  env = new NetraxTest();

  // Set data dir using the program path.
  std::string call = argv[0];
  std::size_t found = call.find_last_of("/\\");
  if (found != std::string::npos) {
      env->data_dir = call.substr(0,found) + "/../data/";
  }

  ::testing::InitGoogleTest(&argc, argv);
//  MPI_INIT(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(env);
  auto result = RUN_ALL_TESTS();
//  MPI_FINALIZE();
  return result;
}
