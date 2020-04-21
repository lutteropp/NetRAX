/*
 * NetworkInfo.cpp
 *
 *  Created on: Apr 21, 2020
 *      Author: sarah
 */

#include "NetworkInfo.hpp"

#include <string>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "io/NetworkIO.hpp"
#include "NetraxOptions.hpp"
#include "RaxmlWrapper.hpp"
#include <raxml-ng/TreeInfo.hpp>

#include "likelihood/LikelihoodComputation.hpp"

namespace netrax {

NetworkInfo buildNetworkInfo(const NetraxOptions &options) {
    NetworkInfo clump;
    clump.network = readNetworkFromFile(options.network_file);
    RaxmlWrapper wrapper(options);
    clump.raxml_treeinfo = wrapper.createRaxmlTreeinfo(clump.network);
    RaxmlWrapper::NetworkParams *params =
            (RaxmlWrapper::NetworkParams*) clump.raxml_treeinfo.pll_treeinfo().likelihood_computation_params;
    clump.treeinfo = *(params->network_treeinfo);
    return clump;
}

NetworkInfo buildNetworkInfo(const std::string &networkPath, const std::string &msaPath, bool useRepeats) {
    NetraxOptions options;
    options.network_file = networkPath;
    options.msa_file = msaPath;
    options.use_repeats = useRepeats;
    return buildNetworkInfo(options);
}

double loglh(NetworkInfo &networkinfo, bool incremental, bool use_blobs, bool use_graycode) {
    return netrax::computeLoglikelihood(networkinfo, incremental, true, false, use_blobs, use_graycode);
}

double update_reticulation_probs(NetworkInfo &networkinfo, bool incremental, bool use_blobs, bool use_graycode) {
    return netrax::computeLoglikelihood(networkinfo, incremental, true, true, use_blobs, use_graycode);
}

double optimize_model(NetworkInfo &networkinfo, double lh_epsilon) {
    return networkinfo.raxml_treeinfo.optimize_model(lh_epsilon);
}

double optimize_branches(NetworkInfo &networkinfo, double lh_epsilon, double brlen_smooth_factor) {
    return networkinfo.raxml_treeinfo.optimize_branches(lh_epsilon, brlen_smooth_factor);
}

}
