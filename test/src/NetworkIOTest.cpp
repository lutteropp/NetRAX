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
#include "src/Network.hpp"

using namespace netrax;

void check_node_types(const Network& network) {
	for (size_t i = 0; i < network.nodes.size(); ++i) {
		if (std::find(network.reticulation_nodes.begin(), network.reticulation_nodes.end(), &network.nodes[i])
				== network.reticulation_nodes.end()) {
			ASSERT_EQ(network.nodes[i].getType(), NodeType::BASIC_NODE);
		} else {
			ASSERT_EQ(network.nodes[i].getType(), NodeType::RETICULATION_NODE);
		}
	}
}

void check_neighbor_count(const Network& network) {
	for (size_t i = 0; i < network.tip_nodes.size(); ++i) {
		ASSERT_EQ(network.tip_nodes[i]->getNeighbors().size(), 1);
	}
	for (size_t i = 0; i < network.inner_nodes.size(); ++i) {
		ASSERT_EQ(network.inner_nodes[i]->getNeighbors().size(), 3);
	}
}

void sanity_checks(const Network& network) {
	check_node_types(network);
	check_neighbor_count(network);
}

TEST (NetworkIOTest, testTheTest) {
	ASSERT_TRUE(true);
}

TEST(NetworkIOTest, convertNetworkTest) {
	std::string newick = "((A:2,((B:1,C:1)P:1)X#H1:0::0.3)Q:2,(D:2,X#H1:0::0.7)R:2);";
	unetwork_t * unetwork = unetwork_parse_newick_string(newick.c_str());
	Network network = convertNetwork(*unetwork);

	ASSERT_EQ(network.root->getLink()->index, unetwork->vroot->node_index);
	size_t n = unetwork->tip_count + unetwork->inner_tree_count + unetwork->reticulation_count;
	for (size_t i = 0; i < n; ++i) {
		size_t clv_idx = unetwork->nodes[i]->clv_index;
		ASSERT_EQ(unetwork->nodes[i]->node_index, network.nodes[clv_idx].getLink()->index);
		ASSERT_EQ(unetwork->nodes[i]->pmatrix_index, network.links[unetwork->nodes[i]->node_index].edge->getIndex());
		ASSERT_EQ(unetwork->nodes[i]->scaler_index, network.nodes[clv_idx].getScalerIndex());
		ASSERT_EQ(clv_idx, network.nodes[clv_idx].getIndex());
	}
	ASSERT_EQ(unetwork->reticulation_count, network.num_reticulations());
	ASSERT_EQ(unetwork->edge_count, network.num_branches());
	ASSERT_EQ(unetwork->inner_tree_count, network.num_inner() - network.num_reticulations());
	ASSERT_EQ(unetwork->tip_count, network.num_tips());
}

TEST(NetworkIOTest, readRootedTree) {
	std::string input = "((A:0.1,B:0.2):0.1,(C:0.3,D:0.4):0.5);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readUnrootedTree) {
	std::string input = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetwork) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0::0.3)Q:2,(D:2,X#H1:0::0.7)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	ASSERT_EQ(1, network.num_reticulations());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetwork2) {
	std::string input = "((A,(B)x#H1),(D,x#H1));";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(3, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationNoExtra) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1)Q:2,(D:2,X#H1)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationNoExtraNoLabel) {
	std::string input = "((A:2,((B:1,C:1)P:1)#H1)Q:2,(D:2,#H1)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationOnlyLength) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0)Q:2,(D:2,X#H1:0)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationOnlyProb) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:::0.3)Q:2,(D:2,X#H1:::0.7)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationOnlySupport) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1::0)Q:2,(D:2,X#H1::0)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationLengthAndSupport) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0:0)Q:2,(D:2,X#H1:0:0)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationLengthAndProb) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0::1)Q:2,(D:2,X#H1:0::0)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationSupportAndProb) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1::0:1)Q:2,(D:2,X#H1::0:0)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkLowercaseTaxa) {
	std::string input = "((a:2,((b:1,c:1)P:1)X#H1:0::0.3)Q:2,(d:2,X#H1:0::0.7)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkLowercaseTaxaAndInternalTree) {
	std::string input = "((a:2,((b:1,c:1)p:1)X#H1:0::0.3)q:2,(d:2,X#H1:0::0.7)r:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkLowercaseLabels) {
	std::string input = "((a:2,((b:1,c:1)p:1)x#H1:0::0.3)q:2,(d:2,x#H1:0::0.7)r:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkLowercaseAll) {
	std::string input = "((a:2,((b:1,c:1)p:1)x#h1:0::0.3)q:2,(d:2,x#h1:0::0.7)r:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readCelineExample1Simplified) {
	std::string input = "((((Anolis,(Podarcis)#H1),(((#H1,Taeniopygia),Caiman),(Emys,(Chelonoidi,(Caretta)#H2)))),(#H2,Monodelphis)));";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(8, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readCelineNetwork) {
	std::string input =
			"((protopterus:0.0,(Xenopus:0.0,(((((Monodelphis:0.0,(python:0.0)#H1:0.0):0.0,(Caretta:0.0)#H2:0.0):0.0,(Homo:0.0)#H3:0.0):0.0,(Ornithorhynchus:0.0)#H4:0.0):0.0,(((#H1:0.0,((#H3:0.0,Anolis:0.0):0.0,(Gallus:0.0)#H5:0.0):0.0):0.0,(Podarcis:0.0)#H6:0.0):0.0,(((#H5:0.0,(#H6:0.0,Taeniopygia:0.0):0.0):0.0,(alligator:0.0,Caiman:0.0):0.0):0.0,(phrynops:0.0,(Emys:0.0,((Chelonoidi:0.0,#H4:0.0):0.0,#H2:0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(16, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkCelineStyle0) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)X#H1:0.0::0.3)Q:0.0,(D:0.0,X#H1:0.0::0.7)R:0.0);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkCelineStyle2) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)x#H1:0.0:0.0:0.0)Q:0.0,(D:0.0,x#H1:0.0:0.0:0.0)R:0.0);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkCelineStyle3) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)x#H1:0.0)Q:0.0,(D:0.0,x#H1:0.0)R:0.0);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkCelineStyle) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)#H1:0.0)Q:0.0,(D:0.0,#H1:0.0)R:0.0);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkCelineStyleNoLengths) {
	std::string input = "((A,((B,C)P)#H1)Q,(D,#H1)R);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, celineExample1x) {
	std::string input = "((((((A,(P)X#H1),(((X#H1,T),(A2,Caiman)),(P2,(E,(C1,(C2)Y#H2))))),(Y#H2,M)),X2),P2));";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(12, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, celineExample1) {
	std::string input =
			"((((((Anolis,(Podarcis)#H1),(((#H1,Taeniopygia),(alligator,Caiman)),(phrynops,(Emys,(Chelonoidi,(Caretta)#H2))))),(#H2,Monodelphis)),Xenopus),protopterus));";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(12, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, celineExample2) {
	std::string input =
			"(((((((Anolis,(Gallus)#H1),(Podarcis)#H2),(((#H1,(#H2,Taeniopygia)),(alligator,Caiman)),(phrynops,(Emys,((Chelonoidi,(Ornithorhynchus)#H3),(Caretta)#H4))))),(#H3,(#H4,Monodelphis))),Xenopus),protopterus));";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(14, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, celineExample3) {
	std::string input =
			"((protopterus,(Xenopus,(((Monodelphis,(Caretta)#H1),(Homo)#H2),(((#H2,Anolis),(Podarcis)#H3),(((#H3,Taeniopygia),(alligator,Caiman)),(phrynops,(Emys,(Chelonoidi,#H1)))))))));";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(13, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkCelineStyle1WithSupportValues) {
	std::string input = "((A:0.0,((B:0.0,C:0.0)P:0.0)X#H1:0.0:0.0:0.3)Q:0.0,(D:0.0,X#H1:0.0:0.0:0.7)R:0.0);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}

TEST(NetworkIOTest, readSimpleNetworkReticulationLengthAndProbAndSupport) {
	std::string input = "((A:2,((B:1,C:1)P:1)X#H1:0:0:1)Q:2,(D:2,X#H1:0:0:0)R:2);";
	Network network = readNetworkFromString(input);
	ASSERT_EQ(4, network.num_tips());
	sanity_checks(network);
}
