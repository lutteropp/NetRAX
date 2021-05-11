#pragma once

#include "GeneralMove.hpp"
struct Edge;

namespace netrax {

struct RSPRMove: public GeneralMove {
    RSPRMove(size_t edge_orig_idx) :
            GeneralMove(MoveType::RSPRMove, edge_orig_idx) {
    }

    RSPRMove() :
            GeneralMove(MoveType::RSPRMove, 0) {
    }
    
    size_t x_prime_clv_index = 0;
    size_t y_prime_clv_index = 0;
    size_t x_clv_index = 0;
    size_t y_clv_index = 0;
    size_t z_clv_index = 0;

    std::vector<double> x_z_len = {0};
    std::vector<double> z_y_len = {0};
    std::vector<double> x_prime_y_prime_len = {0};

    RSPRMove(RSPRMove&& rhs) : GeneralMove{rhs}, x_prime_clv_index{rhs.x_prime_clv_index}, y_prime_clv_index{rhs.y_prime_clv_index}, x_clv_index{rhs.x_clv_index}, y_clv_index{rhs.y_clv_index}, z_clv_index{rhs.z_clv_index}, x_z_len{rhs.x_z_len}, z_y_len{rhs.z_y_len}, x_prime_y_prime_len{rhs.x_prime_y_prime_len} {}

    RSPRMove(const RSPRMove& rhs) : GeneralMove{rhs}, x_prime_clv_index{rhs.x_prime_clv_index}, y_prime_clv_index{rhs.y_prime_clv_index}, x_clv_index{rhs.x_clv_index}, y_clv_index{rhs.y_clv_index}, z_clv_index{rhs.z_clv_index}, x_z_len{rhs.x_z_len}, z_y_len{rhs.z_y_len}, x_prime_y_prime_len{rhs.x_prime_y_prime_len} {}

    RSPRMove& operator =(RSPRMove&& rhs)
    {
        if (this != &rhs)
        {
            x_prime_clv_index = rhs.x_prime_clv_index;
            y_prime_clv_index = rhs.y_prime_clv_index;
            x_clv_index = rhs.x_clv_index;
            y_clv_index = rhs.y_clv_index;
            z_clv_index = rhs.z_clv_index;
            x_z_len = rhs.x_z_len;
            z_y_len = rhs.z_y_len;
            x_prime_y_prime_len = rhs.x_prime_y_prime_len;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }

    RSPRMove& operator =(const RSPRMove& rhs)
    {
        if (this != &rhs)
        {
            x_prime_clv_index = rhs.x_prime_clv_index;
            y_prime_clv_index = rhs.y_prime_clv_index;
            x_clv_index = rhs.x_clv_index;
            y_clv_index = rhs.y_clv_index;
            z_clv_index = rhs.z_clv_index;
            x_z_len = rhs.x_z_len;
            z_y_len = rhs.z_y_len;
            x_prime_y_prime_len = rhs.x_prime_y_prime_len;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }
};

bool checkSanity(AnnotatedNetwork& ann_network, RSPRMove& move);

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network);

std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves = false);

std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Node *node, bool noRSPR1Moves = false, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, const Node *node, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());
std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, const Node *node, bool noRSPR1Moves = false, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, const Node *node, bool noRSPR1Moves = false, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());

void performMove(AnnotatedNetwork &ann_network, RSPRMove &move);
void undoMove(AnnotatedNetwork &ann_network, RSPRMove &move);

std::string toString(RSPRMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RSPRMove &move);

std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RSPRMove &move);
RSPRMove randomRSPRMove(AnnotatedNetwork &ann_network);
RSPRMove randomRSPR1Move(AnnotatedNetwork &ann_network);
RSPRMove randomTailMove(AnnotatedNetwork &ann_network);
RSPRMove randomHeadMove(AnnotatedNetwork &ann_network);

inline bool needsRecompute(AnnotatedNetwork& ann_network, const RSPRMove& move) {
    return false;
}

}