/*
 * NetworkIOTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/io/NetworkIO.hpp"

#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include "src/Network.hpp"

using namespace netrax;

TEST (NetworkIOTest, testTheTest) {
	ASSERT_TRUE(true);
}

TEST(NetworkIOTest, readRootedTree) {
	std::string input = "((A:0.1,B:0.2):0.1,(C:0.3,D:0.4):0.5);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.tip_count);
}
