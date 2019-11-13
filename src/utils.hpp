/*
 * utils.hpp
 *
 *  Created on: Nov 13, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <iostream>
#include <vector>

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
#include <libpll/pllmod_util.h>
}

namespace netrax {

void print_model_params(const pll_partition_t* partition);

}
