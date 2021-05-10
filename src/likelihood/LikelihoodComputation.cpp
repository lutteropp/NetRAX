/*
 * LikelihoodComputation.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: Sarah Lutteropp
 */

#include "LikelihoodComputation.hpp"
#include "../helper/Helper.hpp"

#include "PseudoLoglikelihood.hpp"
#include "ImprovedLoglikelihood.hpp"

namespace netrax {

double computeLoglikelihood(AnnotatedNetwork &ann_network, int incremental, int update_pmatrices) {
    //just for debug
    //incremental = 0;
    //update_pmatrices = 1;
    if (ann_network.options.likelihood_variant == LikelihoodVariant::SARAH_PSEUDO) {
        return computePseudoLoglikelihood(ann_network, incremental, update_pmatrices);
    } else {
        return computeLoglikelihoodImproved(ann_network, incremental, update_pmatrices);
    }
    //return computeLoglikelihoodNaiveUtree(ann_network, incremental, update_pmatrices);
}

}
