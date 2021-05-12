#pragma once

#include "GeneralMove.hpp"
struct Edge;

namespace netrax {

struct ArcInsertionMove: public GeneralMove {
    ArcInsertionMove(size_t edge_orig_idx, size_t node_orig_idx) :
        GeneralMove(MoveType::ArcInsertionMove, edge_orig_idx, node_orig_idx) {
    }

    ArcInsertionMove() :
        GeneralMove(MoveType::ArcInsertionMove, 0, 0) {
    }

    size_t a_clv_index = 0;
    size_t b_clv_index = 0;
    size_t c_clv_index = 0;
    size_t d_clv_index = 0;

    std::vector<double> u_v_len = {1.0};
    std::vector<double> c_v_len = {1.0};
    std::vector<double> a_u_len = {1.0};

    size_t wanted_u_clv_index = std::numeric_limits<size_t>::max();
    size_t wanted_v_clv_index = std::numeric_limits<size_t>::max();
    size_t wanted_au_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_ub_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_cv_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_vd_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t wanted_uv_pmatrix_index = std::numeric_limits<size_t>::max();

    size_t ab_pmatrix_index = std::numeric_limits<size_t>::max();
    size_t cd_pmatrix_index = std::numeric_limits<size_t>::max();
    std::vector<double> a_b_len = {0};
    std::vector<double> c_d_len = {0};

    std::vector<double> v_d_len = {0};
    std::vector<double> u_b_len = {0};

    ArcInsertionMove(ArcInsertionMove&& rhs) : GeneralMove{rhs}, a_clv_index{rhs.a_clv_index}, b_clv_index{rhs.b_clv_index}, c_clv_index{rhs.c_clv_index}, d_clv_index{rhs.d_clv_index}, u_v_len{rhs.u_v_len}, c_v_len{rhs.c_v_len}, a_u_len{rhs.a_u_len}, wanted_u_clv_index{rhs.wanted_u_clv_index}, wanted_v_clv_index{rhs.wanted_v_clv_index}, wanted_au_pmatrix_index{rhs.wanted_au_pmatrix_index}, wanted_ub_pmatrix_index{rhs.wanted_ub_pmatrix_index}, wanted_cv_pmatrix_index{rhs.wanted_cv_pmatrix_index}, wanted_vd_pmatrix_index{rhs.wanted_vd_pmatrix_index}, wanted_uv_pmatrix_index{rhs.wanted_uv_pmatrix_index}, ab_pmatrix_index{rhs.ab_pmatrix_index}, cd_pmatrix_index{rhs.cd_pmatrix_index}, a_b_len{rhs.a_b_len}, c_d_len{rhs.c_d_len}, v_d_len{rhs.v_d_len}, u_b_len{rhs.u_b_len}  {}

    ArcInsertionMove(const ArcInsertionMove& rhs) : GeneralMove{rhs}, a_clv_index{rhs.a_clv_index}, b_clv_index{rhs.b_clv_index}, c_clv_index{rhs.c_clv_index}, d_clv_index{rhs.d_clv_index}, u_v_len{rhs.u_v_len}, c_v_len{rhs.c_v_len}, a_u_len{rhs.a_u_len}, wanted_u_clv_index{rhs.wanted_u_clv_index}, wanted_v_clv_index{rhs.wanted_v_clv_index}, wanted_au_pmatrix_index{rhs.wanted_au_pmatrix_index}, wanted_ub_pmatrix_index{rhs.wanted_ub_pmatrix_index}, wanted_cv_pmatrix_index{rhs.wanted_cv_pmatrix_index}, wanted_vd_pmatrix_index{rhs.wanted_vd_pmatrix_index}, wanted_uv_pmatrix_index{rhs.wanted_uv_pmatrix_index}, ab_pmatrix_index{rhs.ab_pmatrix_index}, cd_pmatrix_index{rhs.cd_pmatrix_index}, a_b_len{rhs.a_b_len}, c_d_len{rhs.c_d_len}, v_d_len{rhs.v_d_len}, u_b_len{rhs.u_b_len}  {}

    ArcInsertionMove& operator =(ArcInsertionMove&& rhs)
    {
        if (this != &rhs)
        {
            a_clv_index = rhs.a_clv_index;
            b_clv_index = rhs.b_clv_index;
            c_clv_index = rhs.c_clv_index;
            d_clv_index = rhs.d_clv_index;
            u_v_len = rhs.u_v_len;
            c_v_len = rhs.c_v_len;
            a_u_len = rhs.a_u_len;
            wanted_u_clv_index = rhs.wanted_u_clv_index;
            wanted_v_clv_index = rhs.wanted_v_clv_index;
            wanted_au_pmatrix_index = rhs.wanted_au_pmatrix_index;
            wanted_ub_pmatrix_index = rhs.wanted_ub_pmatrix_index;
            wanted_cv_pmatrix_index = rhs.wanted_cv_pmatrix_index;
            wanted_vd_pmatrix_index = rhs.wanted_vd_pmatrix_index;
            wanted_uv_pmatrix_index = rhs.wanted_uv_pmatrix_index;
            ab_pmatrix_index = rhs.ab_pmatrix_index;
            cd_pmatrix_index = rhs.cd_pmatrix_index;
            a_b_len = rhs.a_b_len;
            c_d_len = rhs.c_d_len;
            v_d_len = rhs.v_d_len;
            u_b_len = rhs.u_b_len;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }

    ArcInsertionMove& operator =(const ArcInsertionMove& rhs)
    {
        if (this != &rhs)
        {
            a_clv_index = rhs.a_clv_index;
            b_clv_index = rhs.b_clv_index;
            c_clv_index = rhs.c_clv_index;
            d_clv_index = rhs.d_clv_index;
            u_v_len = rhs.u_v_len;
            c_v_len = rhs.c_v_len;
            a_u_len = rhs.a_u_len;
            wanted_u_clv_index = rhs.wanted_u_clv_index;
            wanted_v_clv_index = rhs.wanted_v_clv_index;
            wanted_au_pmatrix_index = rhs.wanted_au_pmatrix_index;
            wanted_ub_pmatrix_index = rhs.wanted_ub_pmatrix_index;
            wanted_cv_pmatrix_index = rhs.wanted_cv_pmatrix_index;
            wanted_vd_pmatrix_index = rhs.wanted_vd_pmatrix_index;
            wanted_uv_pmatrix_index = rhs.wanted_uv_pmatrix_index;
            ab_pmatrix_index = rhs.ab_pmatrix_index;
            cd_pmatrix_index = rhs.cd_pmatrix_index;
            a_b_len = rhs.a_b_len;
            c_d_len = rhs.c_d_len;
            v_d_len = rhs.v_d_len;
            u_b_len = rhs.u_b_len;
        }
        GeneralMove::operator=(rhs);
        return *this;
    }
};

bool checkSanity(AnnotatedNetwork& ann_network, ArcInsertionMove& move);

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const Edge *edge, bool noDeltaPlus = false);
std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, bool noDeltaPlus = false, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network,
        const Edge *edge);
std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());


std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const Node *node, bool noDeltaPlus = false, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());
std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network,
        const Node *node, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const std::vector<Node*>& start_nodes, bool noDeltaPlus = false, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());
std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network,
        const std::vector<Node*>& start_nodes, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());

std::vector<ArcInsertionMove> possibleMoves(AnnotatedNetwork& ann_network, const std::vector<Node*>& start_nodes, ArcInsertionMove placeholderMove, size_t min_radius = 0, size_t max_radius = std::numeric_limits<size_t>::max());

void performMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);

void undoMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);
std::string toString(ArcInsertionMove &move);

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move);

std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move);

ArcInsertionMove randomArcInsertionMove(AnnotatedNetwork &ann_network);
ArcInsertionMove randomDeltaPlusMove(AnnotatedNetwork &ann_network);

ArcInsertionMove buildArcInsertionMove(size_t a_clv_index, size_t b_clv_index, size_t c_clv_index,
        size_t d_clv_index, std::vector<double> &u_v_len, std::vector<double> &c_v_len,
        std::vector<double> &a_u_len, std::vector<double> &a_b_len, std::vector<double> &c_d_len, std::vector<double> &v_d_len, std::vector<double> &u_b_len, MoveType moveType, size_t edge_orig_idx, size_t node_orig_idx);

inline bool needsRecompute(AnnotatedNetwork& ann_network, const ArcInsertionMove& move) {
    return false;
}

}