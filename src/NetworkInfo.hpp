/*
 * NetworkInfo.hpp
 *
 *  Created on: Apr 21, 2020
 *      Author: sarah
 */

#pragma once

#include "graph/Network.hpp"
#include "NetraxOptions.hpp"
#include <raxml-ng/constants.hpp>
#include <raxml-ng/TreeInfo.hpp>

#include <string>

namespace netrax {

struct NetworkInfo {
    Network network;
    pllmod_treeinfo_t treeinfo;
    TreeInfo raxml_treeinfo;
    bool pmatrix_is_dirty = true;
};

NetworkInfo buildNetworkInfo(const NetraxOptions &options);
NetworkInfo buildNetworkInfo(const std::string &networkPath, const std::string &msaPath, bool useRepeats);

double loglh(NetworkInfo &networkinfo, bool incremental = false, bool use_blob_optimization = true,
        bool use_graycode_optimization = true);
double update_reticulation_probs(NetworkInfo &networkinfo, bool incremental = false, bool use_blob_optimization = true,
        bool use_graycode_optimization = true);
double optimize_model(NetworkInfo &networkinfo, double lh_epsilon = DEF_LH_EPSILON);
double optimize_branches(NetworkInfo &networkinfo, double lh_epsilon = DEF_LH_EPSILON, double brlen_smooth_factor = 1);

}
