#include "Scrambling.hpp"

#include "../moves/Moves.hpp"
#include "../optimization/Optimization.hpp"

namespace netrax {

void scrambleNetwork(AnnotatedNetwork& ann_network, MoveType type, size_t scramble_cnt) {
    // perform scramble_cnt moves of the specified move type on the network
    ArcInsertionMove insertionMove(0);
    ArcRemovalMove removalMove(0);
    RNNIMove rnniMove(0);
    RSPRMove rsprMove(0);
    for (size_t i = 0; i < scramble_cnt; ++i) {
        switch (type) {
        case MoveType::ArcInsertionMove:
            insertionMove = randomArcInsertionMove(ann_network);
            performMove(ann_network, insertionMove);
            ann_network.last_accepted_move_edge_orig_idx = insertionMove.edge_orig_idx;
            break;
        case MoveType::ArcRemovalMove:
            removalMove = randomArcRemovalMove(ann_network);
            performMove(ann_network, removalMove);
            ann_network.last_accepted_move_edge_orig_idx = removalMove.edge_orig_idx;
            break;
        case MoveType::DeltaMinusMove:
            removalMove = randomDeltaMinusMove(ann_network);
            performMove(ann_network, removalMove);
            ann_network.last_accepted_move_edge_orig_idx = removalMove.edge_orig_idx;
            break;
        case MoveType::DeltaPlusMove:
            insertionMove = randomDeltaPlusMove(ann_network);
            performMove(ann_network, insertionMove);
            ann_network.last_accepted_move_edge_orig_idx = insertionMove.edge_orig_idx;
            break;
        case MoveType::HeadMove:
            rsprMove = randomHeadMove(ann_network);
            performMove(ann_network, rsprMove);
            ann_network.last_accepted_move_edge_orig_idx = rsprMove.edge_orig_idx;
            break;
        case MoveType::RNNIMove:
            rnniMove = randomRNNIMove(ann_network);
            performMove(ann_network, rnniMove);
            ann_network.last_accepted_move_edge_orig_idx = rnniMove.edge_orig_idx;
            break;
        case MoveType::RSPR1Move:
            rsprMove = randomRSPR1Move(ann_network);
            performMove(ann_network, rsprMove);
            ann_network.last_accepted_move_edge_orig_idx = rsprMove.edge_orig_idx;
            break;
        case MoveType::RSPRMove:
            rsprMove = randomRSPRMove(ann_network);
            performMove(ann_network, rsprMove);
            ann_network.last_accepted_move_edge_orig_idx = rsprMove.edge_orig_idx;
            break;
        case MoveType::TailMove:
            rsprMove = randomTailMove(ann_network);
            performMove(ann_network, rsprMove);
            ann_network.last_accepted_move_edge_orig_idx = rsprMove.edge_orig_idx;
            break;
        default:
            throw std::runtime_error("Unrecognized move type");
        }
    }
    optimizeAllNonTopology(ann_network);
}

}