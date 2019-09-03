/*
 * NetworkIOTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/io/NetworkIO.hpp"

#include <gtest/gtest.h>
#include <string>
#include "src/Network.hpp"

TEST (NetworkIOTest, testTheTest) {
	ASSERT_TRUE(true);
}

TEST(NetworkIOTest, readSimpleTree) {
	std::string input = "(A,B);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(2, network.tip_count);
}
