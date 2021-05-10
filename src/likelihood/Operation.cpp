#include "Operation.hpp"

#include "../helper/Helper.hpp"

namespace netrax {

pll_operation_t buildOperation(Network &network, Node *parent, Node *child1, Node *child2,
        size_t fake_clv_index, size_t fake_pmatrix_index) {
    pll_operation_t operation;
    assert(parent);
    operation.parent_clv_index = parent->clv_index;
    operation.parent_scaler_index = parent->scaler_index;
    if (child1) {
        operation.child1_clv_index = child1->clv_index;
        operation.child1_scaler_index = child1->scaler_index;
        operation.child1_matrix_index = getEdgeTo(network, child1, parent)->pmatrix_index;
    } else {
        operation.child1_clv_index = fake_clv_index;
        operation.child1_scaler_index = -1;
        operation.child1_matrix_index = fake_pmatrix_index;
    }
    if (child2) {
        operation.child2_clv_index = child2->clv_index;
        operation.child2_scaler_index = child2->scaler_index;
        operation.child2_matrix_index = getEdgeTo(network, child2, parent)->pmatrix_index;
    } else {
        operation.child2_clv_index = fake_clv_index;
        operation.child2_scaler_index = -1;
        operation.child2_matrix_index = fake_pmatrix_index;
    }
    return operation;
}

void printOperation(pll_operation_t& op) {
    std::cout << "parent_clv_index: " << op.parent_clv_index << "\n";
    std::cout << "parent_scaler_index: " << op.parent_scaler_index << "\n";
    std::cout << "child1_clv_index: " << op.child1_clv_index << "\n";
    std::cout << "child1_scaler_index: " << op.child1_scaler_index << "\n";
    std::cout << "child1_matrix_index: " << op.child1_matrix_index << "\n";
    std::cout << "child2_clv_index: " << op.child1_clv_index << "\n";
    std::cout << "child2_scaler_index: " << op.child1_scaler_index << "\n";
    std::cout << "child2_matrix_index: " << op.child1_matrix_index << "\n";
}

}