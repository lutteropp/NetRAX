#pragma once

#include "GeneralMove.hpp"
struct Edge;

namespace netrax {

enum class RNNIMoveType {
    ONE, ONE_STAR, TWO, TWO_STAR, THREE, THREE_STAR, FOUR
};

struct RNNIMove: public GeneralMove {
    RNNIMove(size_t edge_orig_idx) :
            GeneralMove(MoveType::RNNIMove, edge_orig_idx) {
    }

    RNNIMove() :
            GeneralMove(MoveType::RNNIMove, 0) {
    }

    size_t u_clv_index = 0;
    size_t v_clv_index = 0;
    size_t s_clv_index = 0;
    size_t t_clv_index = 0;
    RNNIMoveType type = RNNIMoveType::ONE;

    RNNIMove(RNNIMove&& rhs) : GeneralMove{rhs}, u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, s_clv_index{rhs.s_clv_index}, t_clv_index{rhs.t_clv_index}, type{rhs.type} {}

    RNNIMove(const RNNIMove& rhs) : GeneralMove{rhs}, u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, s_clv_index{rhs.s_clv_index}, t_clv_index{rhs.t_clv_index}, type{rhs.type} {}

    RNNIMove& operator =(RNNIMove&& rhs)
    {
        if (this != &rhs)
        {
            u_clv_index = rhs.u_clv_index;
            v_clv_index = rhs.v_clv_index;
            s_clv_index = rhs.s_clv_index;
            t_clv_index = rhs.t_clv_index;
            type = rhs.type;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }

    RNNIMove& operator =(const RNNIMove& rhs)
    {
        if (this != &rhs)
        {
            u_clv_index = rhs.u_clv_index;
            v_clv_index = rhs.v_clv_index;
            s_clv_index = rhs.s_clv_index;
            t_clv_index = rhs.t_clv_index;
            type = rhs.type;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }
};

bool checkSanity(AnnotatedNetwork& ann_network, RNNIMove& move);
std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network);
void performMove(AnnotatedNetwork &ann_network, RNNIMove &move);
void undoMove(AnnotatedNetwork &ann_network, RNNIMove &move);
std::string toString(RNNIMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RNNIMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RNNIMove &move);
RNNIMove randomRNNIMove(AnnotatedNetwork &ann_network);

}
