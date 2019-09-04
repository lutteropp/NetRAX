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

TEST(NetworkIOTest, readUnrootedTree) {
	std::string input = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.tip_count);
}

TEST(NetworkIOTest, readSimpleNetwork) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0::0.3)Q:2,(D:2,X#H1:0::0.7)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.tip_count);
}

TEST(NetworkIOTest, readCelineNetwork) {
	std::string input =
			"((protopterus:0.0,(Xenopus:0.0,(((((Monodelphis:0.0,(python:0.0)#H1:0.0):0.0,(Caretta:0.0)#H2:0.0):0.0,(Homo:0.0)#H3:0.0):0.0,(Ornithorhynchus:0.0)#H4:0.0):0.0,(((#H1:0.0,((#H3:0.0,Anolis:0.0):0.0,(Gallus:0.0)#H5:0.0):0.0):0.0,(Podarcis:0.0)#H6:0.0):0.0,(((#H5:0.0,(#H6:0.0,Taeniopygia:0.0):0.0):0.0,(alligator:0.0,Caiman:0.0):0.0):0.0,(phrynops:0.0,(Emys:0.0,((Chelonoidi:0.0,#H4:0.0):0.0,#H2:0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(16, network.tip_count);
}
