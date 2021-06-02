#pragma once

extern "C" {
#include <libpll/pll.h>
#include <libpll/pll_tree.h>
}

#include "../graph/Network.hpp"

namespace netrax {

pll_operation_t buildOperation(Network &network, Node *parent, Node *child1,
                               Node *child2, size_t fake_clv_index,
                               size_t fake_pmatrix_index);

void printOperation(pll_operation_t &op);

}  // namespace netrax