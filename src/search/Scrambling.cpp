#include "Scrambling.hpp"

#include "../moves/Move.hpp"
#include "../optimization/Optimization.hpp"

namespace netrax {

void scrambleNetwork(AnnotatedNetwork& ann_network, MoveType type, size_t scramble_cnt) {
    // perform scramble_cnt moves of the specified move type on the network
    for (size_t i = 0; i < scramble_cnt; ++i) {
        Move move = randomMove(ann_network, type);
        performMove(ann_network, move);
        ann_network.last_accepted_move_edge_orig_idx = move.edge_orig_idx;
    }
    optimizeAllNonTopology(ann_network);
}

}