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
#include <algorithm>
#include "src/graph/Common.hpp"

#include "NetraxTest.hpp"

#include "src/io/RootedNetworkParser.hpp"

using namespace netrax;

class NetworkIOTest: public ::testing::Test {
protected:
	std::string treePath = "examples/sample_networks/tree.nw";
	std::string networkPath = "examples/sample_networks/small.nw";
	std::string msaPath = "examples/sample_networks/small_fake_alignment.nw";

	virtual void SetUp() {
		g_singleThread.lock();
	}

	virtual void TearDown() {
		g_singleThread.unlock();
	}
};

void check_node_types(const Network &network) {
	for (size_t i = 0; i < network.nodes.size(); ++i) {
		if (std::find(network.reticulation_nodes.begin(), network.reticulation_nodes.end(), &network.nodes[i])
				== network.reticulation_nodes.end()) {
			EXPECT_EQ(network.nodes[i].getType(), NodeType::BASIC_NODE);
		} else {
			EXPECT_EQ(network.nodes[i].getType(), NodeType::RETICULATION_NODE);
		}
	}
}

void check_neighbor_count(const Network &network) {
	for (size_t i = 0; i < network.tip_nodes.size(); ++i) {
		EXPECT_EQ(network.tip_nodes[i]->getLink()->next, nullptr);
		EXPECT_EQ(network.tip_nodes[i]->getNeighbors().size(), 1);
	}
	for (size_t i = 0; i < network.inner_nodes.size(); ++i) {
		EXPECT_NE(network.inner_nodes[i]->getLink()->next, nullptr);
		EXPECT_NE(network.inner_nodes[i]->getLink()->next->next, nullptr);
		EXPECT_NE(network.inner_nodes[i]->getLink()->next->next->next, nullptr);
		EXPECT_EQ(network.inner_nodes[i]->getLink(), network.inner_nodes[i]->getLink()->next->next->next);
		EXPECT_EQ(network.inner_nodes[i]->getNeighbors().size(), 3);
	}
}

void check_tip_clvs(const Network &network) {
	size_t n = network.num_tips();
	EXPECT_EQ(network.tip_nodes.size(), n);
	for (size_t i = 0; i < network.tip_nodes.size(); ++i) {
		EXPECT_TRUE(network.tip_nodes[i]->getClvIndex() < n);
	}
}

void check_pmatrix_indices(const Network &network) {
	std::vector<size_t> allPmatrixIndices;
	for (size_t i = 0; i < network.edges.size(); ++i) {
		allPmatrixIndices.push_back(network.edges[i].pmatrix_index);
	}
	std::sort(allPmatrixIndices.begin(), allPmatrixIndices.end());
	for (size_t i = 0; i < allPmatrixIndices.size(); ++i) {
		EXPECT_EQ(allPmatrixIndices[i], i);
	}
}

void check_links_edges(const Network &network) {
	for (size_t i = 0; i < network.edges.size(); ++i) {
		EXPECT_NE(network.edges[i].link1, nullptr);
		EXPECT_NE(network.edges[i].link2, nullptr);
	}
}

void check_clv_range(const Network &network) {
	std::vector<size_t> allCLVmatrixIndices;
	for (size_t i = 0; i < network.nodes.size(); ++i) {
		allCLVmatrixIndices.push_back(network.nodes[i].clv_index);
	}
	std::sort(allCLVmatrixIndices.begin(), allCLVmatrixIndices.end());
	for (size_t i = 0; i < allCLVmatrixIndices.size(); ++i) {
		EXPECT_EQ(allCLVmatrixIndices[i], i);
	}
}

void sanity_checks(const Network &network) {
	check_node_types(network);
	check_neighbor_count(network);
	check_tip_clvs(network);
	check_links_edges(network);
	check_pmatrix_indices(network);
	check_clv_range(network);
}

TEST_F (NetworkIOTest, testTheTest) {
	EXPECT_TRUE(true);
}

TEST_F (NetworkIOTest, rootedNetworkParserSmall) {
	std::string newick = "((A:2,((B:1,C:1)P:1)X#H1:0::0.3)Q:2,(D:2,X#H1:0::0.7)R:2);";
	RootedNetwork *small = netrax::parseRootedNetworkFromNewickString(newick);
	std::cout << netrax::toNewickString(*small) << "\n";
}

TEST_F (NetworkIOTest, readNetworkFromFile) {
	Network small = readNetworkFromFile(networkPath);
	EXPECT_TRUE(true);
}

TEST_F (NetworkIOTest, readRootedTree) {
	std::string input = "((A:0.1,B:0.2):0.1,(C:0.3,D:0.4):0.5);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readTinyNetwork) {
	std::string input = "((A:2,(B:1)X#H1:0::0.3)Q:2,(D:2,X#H1:0::0.7)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(3, network.num_tips());
	EXPECT_EQ(7, network.num_nodes());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readUnrootedTree) {
	std::string input = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetwork) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0::0.3)Q:2,(D:2,X#H1:0::0.7)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	EXPECT_EQ(1, network.num_reticulations());
	sanity_checks(network);
	std::cout << toExtendedNewick(network) << "\n";
}

TEST_F (NetworkIOTest, readSimpleNetwork2) {
	std::string input = "((A,(B)x#H1),(D,x#H1));";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(3, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, reticulationHasLeafChild) {
	std::string input = "((A:2,(B:1)X#H1)Q:2,(D:2,X#H1)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(3, network.num_tips());
	sanity_checks(network);
}


TEST_F (NetworkIOTest, readSimpleNetworkReticulationNoExtra) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1)Q:2,(D:2,X#H1)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationNoExtraNoLabel) {
	std::string input = "((A:2,((B:1,C:1)P:1)#H1)Q:2,(D:2,#H1)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationOnlyLength) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0)Q:2,(D:2,X#H1:0)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationOnlyProb) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:::0.3)Q:2,(D:2,X#H1:::0.7)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationOnlySupport) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1::0)Q:2,(D:2,X#H1::0)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationLengthAndSupport) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0:0)Q:2,(D:2,X#H1:0:0)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationLengthAndProb) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0::1)Q:2,(D:2,X#H1:0::0)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationSupportAndProb) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1::0:1)Q:2,(D:2,X#H1::0:0)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkLowercaseTaxa) {
	std::string input = "((a:2,((b:1,c:1)P:1)X#H1:0::0.3)Q:2,(d:2,X#H1:0::0.7)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkLowercaseTaxaAndInternalTree) {
	std::string input = "((a:2,((b:1,c:1)p:1)X#H1:0::0.3)q:2,(d:2,X#H1:0::0.7)r:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkLowercaseLabels) {
	std::string input = "((a:2,((b:1,c:1)p:1)x#H1:0::0.3)q:2,(d:2,x#H1:0::0.7)r:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkLowercaseAll) {
	std::string input = "((a:2,((b:1,c:1)p:1)x#h1:0::0.3)q:2,(d:2,x#h1:0::0.7)r:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readCelineExample1Simplified) {
	std::string input =
			"((((Anolis,(Podarcis)#H1),(((#H1,Taeniopygia),Caiman),(Emys,(Chelonoidi,(Caretta)#H2)))),(#H2,Monodelphis)));";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(8, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readCelineNetwork) {
	std::string input =
			"((protopterus:0.0,(Xenopus:0.0,(((((Monodelphis:0.0,(python:0.0)#H1:0.0):0.0,(Caretta:0.0)#H2:0.0):0.0,(Homo:0.0)#H3:0.0):0.0,(Ornithorhynchus:0.0)#H4:0.0):0.0,(((#H1:0.0,((#H3:0.0,Anolis:0.0):0.0,(Gallus:0.0)#H5:0.0):0.0):0.0,(Podarcis:0.0)#H6:0.0):0.0,(((#H5:0.0,(#H6:0.0,Taeniopygia:0.0):0.0):0.0,(alligator:0.0,Caiman:0.0):0.0):0.0,(phrynops:0.0,(Emys:0.0,((Chelonoidi:0.0,#H4:0.0):0.0,#H2:0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(16, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkCelineStyle0) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)X#H1:0.0::0.3)Q:0.0,(D:0.0,X#H1:0.0::0.7)R:0.0);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkCelineStyle2) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)x#H1:0.0:0.0:0.0)Q:0.0,(D:0.0,x#H1:0.0:0.0:0.0)R:0.0);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkCelineStyle3) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)x#H1:0.0)Q:0.0,(D:0.0,x#H1:0.0)R:0.0);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkCelineStyle) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)#H1:0.0)Q:0.0,(D:0.0,#H1:0.0)R:0.0);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkCelineStyleNoLengths) {
	std::string input = "((A,((B,C)P)#H1)Q,(D,#H1)R);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, celineExample1x) {
	std::string input = "((((((A,(P)X#H1),(((X#H1,T),(A2,Caiman)),(P2,(E,(C1,(C2)Y#H2))))),(Y#H2,M)),X2),P2));";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(12, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, celineExample1) {
	std::string input =
			"((((((Anolis,(Podarcis)#H1),(((#H1,Taeniopygia),(alligator,Caiman)),(phrynops,(Emys,(Chelonoidi,(Caretta)#H2))))),(#H2,Monodelphis)),Xenopus),protopterus));";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(12, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, celineExample2) {
	std::string input =
			"(((((((Anolis,(Gallus)#H1),(Podarcis)#H2),(((#H1,(#H2,Taeniopygia)),(alligator,Caiman)),(phrynops,(Emys,((Chelonoidi,(Ornithorhynchus)#H3),(Caretta)#H4))))),(#H3,(#H4,Monodelphis))),Xenopus),protopterus));";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(14, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, celineExample3) {
	std::string input =
			"((protopterus,(Xenopus,(((Monodelphis,(Caretta)#H1),(Homo)#H2),(((#H2,Anolis),(Podarcis)#H3),(((#H3,Taeniopygia),(alligator,Caiman)),(phrynops,(Emys,(Chelonoidi,#H1)))))))));";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(13, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkCelineStyle1WithSupportValues) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)X#H1:0.0:0.0:0.3)Q:0.0,(D:0.0,X#H1:0.0:0.0:0.7)R:0.0);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST_F (NetworkIOTest, readSimpleNetworkReticulationLengthAndProbAndSupport) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0:0:1)Q:2,(D:2,X#H1:0:0:0)R:2);";
	Network network = readNetworkFromString(input);
	EXPECT_EQ(4, network.num_tips());
	sanity_checks(network);
}
