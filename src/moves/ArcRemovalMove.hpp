#pragma once

#include "GeneralMove.hpp"
struct Node;

namespace netrax {

struct ArcRemovalMove: public GeneralMove {
    ArcRemovalMove(size_t edge_orig_idx) :
        GeneralMove(MoveType::ArcRemovalMove, edge_orig_idx) {
    }

    ArcRemovalMove() :
        GeneralMove(MoveType::ArcRemovalMove, 0) {
    }

    size_t a_clv_index = 0;
    size_t b_clv_index = 0;
    size_t c_clv_index = 0;
    size_t d_clv_index = 0;
    size_t u_clv_index = 0;
    size_t v_clv_index = 0;

    std::vector<double> u_v_len = {0.0};
    std::vector<double> c_v_len = {0.0};
    std::vector<double> a_u_len = {1.0};

    size_t au_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t ub_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t cv_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t vd_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t uv_pmatrix_index = std::numeric_limits<size_t>::max();

    size_t wanted_ab_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_cd_pmatrix_index = std::numeric_limits<size_t>::max();
    std::vector<double> a_b_len = {0};
    std::vector<double> c_d_len = {0};

    std::vector<double> v_d_len = {0};
    std::vector<double> u_b_len = {0};

    ArcRemovalMove(ArcRemovalMove&& rhs) : GeneralMove{rhs}, a_clv_index{rhs.a_clv_index}, b_clv_index{rhs.b_clv_index}, c_clv_index{rhs.c_clv_index}, d_clv_index{rhs.d_clv_index}, u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, u_v_len{rhs.u_v_len}, c_v_len{rhs.c_v_len}, a_u_len{rhs.a_u_len}, au_pmatrix_index{rhs.au_pmatrix_index}, ub_pmatrix_index{rhs.ub_pmatrix_index}, cv_pmatrix_index{rhs.cv_pmatrix_index}, vd_pmatrix_index{rhs.vd_pmatrix_index}, uv_pmatrix_index{rhs.uv_pmatrix_index}, wanted_ab_pmatrix_index{rhs.wanted_ab_pmatrix_index}, wanted_cd_pmatrix_index{rhs.wanted_cd_pmatrix_index}, a_b_len{rhs.a_b_len}, c_d_len{rhs.c_d_len}, v_d_len{rhs.v_d_len}, u_b_len{rhs.u_b_len} {}

    ArcRemovalMove(const ArcRemovalMove& rhs) : GeneralMove{rhs}, a_clv_index{rhs.a_clv_index}, b_clv_index{rhs.b_clv_index}, c_clv_index{rhs.c_clv_index}, d_clv_index{rhs.d_clv_index}, u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, u_v_len{rhs.u_v_len}, c_v_len{rhs.c_v_len}, a_u_len{rhs.a_u_len}, au_pmatrix_index{rhs.au_pmatrix_index}, ub_pmatrix_index{rhs.ub_pmatrix_index}, cv_pmatrix_index{rhs.cv_pmatrix_index}, vd_pmatrix_index{rhs.vd_pmatrix_index}, uv_pmatrix_index{rhs.uv_pmatrix_index}, wanted_ab_pmatrix_index{rhs.wanted_ab_pmatrix_index}, wanted_cd_pmatrix_index{rhs.wanted_cd_pmatrix_index}, a_b_len{rhs.a_b_len}, c_d_len{rhs.c_d_len}, v_d_len{rhs.v_d_len}, u_b_len{rhs.u_b_len} {}

    ArcRemovalMove& operator =(ArcRemovalMove&& rhs)
    {
        if (this != &rhs)
        {
            a_clv_index = rhs.a_clv_index;
            b_clv_index = rhs.b_clv_index;
            c_clv_index = rhs.c_clv_index;
            d_clv_index = rhs.d_clv_index;
            u_clv_index = rhs.u_clv_index;
            v_clv_index = rhs.v_clv_index;
            u_v_len = rhs.u_v_len;
            c_v_len = rhs.c_v_len;
            a_u_len = rhs.a_u_len;
            au_pmatrix_index = rhs.au_pmatrix_index;
            ub_pmatrix_index = rhs.ub_pmatrix_index;
            cv_pmatrix_index = rhs.cv_pmatrix_index;
            vd_pmatrix_index = rhs.vd_pmatrix_index;
            uv_pmatrix_index = rhs.uv_pmatrix_index;
            wanted_ab_pmatrix_index = rhs.wanted_ab_pmatrix_index;
            wanted_cd_pmatrix_index = rhs.wanted_cd_pmatrix_index;
            a_b_len = rhs.a_b_len;
            c_d_len = rhs.c_d_len;
            v_d_len = rhs.v_d_len;
            u_b_len = rhs.u_b_len;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }

    ArcRemovalMove& operator =(const ArcRemovalMove& rhs)
    {
        if (this != &rhs)
        {
            a_clv_index = rhs.a_clv_index;
            b_clv_index = rhs.b_clv_index;
            c_clv_index = rhs.c_clv_index;
            d_clv_index = rhs.d_clv_index;
            u_clv_index = rhs.u_clv_index;
            v_clv_index = rhs.v_clv_index;
            u_v_len = rhs.u_v_len;
            c_v_len = rhs.c_v_len;
            a_u_len = rhs.a_u_len;
            au_pmatrix_index = rhs.au_pmatrix_index;
            ub_pmatrix_index = rhs.ub_pmatrix_index;
            cv_pmatrix_index = rhs.cv_pmatrix_index;
            vd_pmatrix_index = rhs.vd_pmatrix_index;
            uv_pmatrix_index = rhs.uv_pmatrix_index;
            wanted_ab_pmatrix_index = rhs.wanted_ab_pmatrix_index;
            wanted_cd_pmatrix_index = rhs.wanted_cd_pmatrix_index;
            a_b_len = rhs.a_b_len;
            c_d_len = rhs.c_d_len;
            v_d_len = rhs.v_d_len;
            u_b_len = rhs.u_b_len;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }
};

bool checkSanity(AnnotatedNetwork& ann_network, ArcRemovalMove& move);

std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network, Node *v, size_t edge_orig_idx,
        MoveType moveType = MoveType::ArcRemovalMove);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network);

std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network, Node *v, size_t edge_orig_idx);
std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network);

void performMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);
void undoMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);

std::string toString(ArcRemovalMove &move);

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, ArcRemovalMove &move);

std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcRemovalMove &move);

ArcRemovalMove randomArcRemovalMove(AnnotatedNetwork &ann_network);
ArcRemovalMove randomDeltaMinusMove(AnnotatedNetwork &ann_network);

ArcRemovalMove buildArcRemovalMove(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, size_t u_clv_index, size_t v_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
         std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType, size_t edge_orig_idx);

inline bool needsRecompute(AnnotatedNetwork& ann_network, const ArcRemovalMove& move) {
    return (ann_network.network.reticulation_nodes[ann_network.network.num_reticulations() - 1]->clv_index != move.v_clv_index);
}

}