/*
 * NetworkIOTest.cpp
 *
 *  Created on: Sep 3, 2019
 *      Author: Sarah Lutteropp
 */

#include "src/likelihood/LikelihoodComputation.hpp"
#include "src/io/NetworkIO.hpp"
#include "src/Fake.hpp"

#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include "src/Network.hpp"

using namespace netrax;

TEST (LikelihoodTest, testTheTest) {
	ASSERT_TRUE(true);
}

TEST (LikelihoodTest, celineNetwork) {
	std::string input =
				"((protopterus:0.0,(Xenopus:0.0,(((((Monodelphis:0.0,(python:0.0)#H1:0.0):0.0,(Caretta:0.0)#H2:0.0):0.0,(Homo:0.0)#H3:0.0):0.0,(Ornithorhynchus:0.0)#H4:0.0):0.0,(((#H1:0.0,((#H3:0.0,Anolis:0.0):0.0,(Gallus:0.0)#H5:0.0):0.0):0.0,(Podarcis:0.0)#H6:0.0):0.0,(((#H5:0.0,(#H6:0.0,Taeniopygia:0.0):0.0):0.0,(alligator:0.0,Caiman:0.0):0.0):0.0,(phrynops:0.0,(Emys:0.0,((Chelonoidi:0.0,#H4:0.0):0.0,#H2:0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0):0.0);";
	Network network = readNetworkFromString(input);
	Options raxml_opts;
	PartitionedMSA parted_msa;
	IDVector tip_msa_idmap;
	PartitionAssignment part_assign;

	TreeInfo raxml_treeinfo = create_fake_raxml_treeinfo(network, raxml_opts, parted_msa, tip_msa_idmap, part_assign);
	double network_logl = raxml_treeinfo.loglh(false);
	std::cout << "The computed network_logl is: " << network_logl << "\n";
	ASSERT_TRUE(true);
}
