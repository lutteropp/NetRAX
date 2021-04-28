/*
 * Moves.hpp
 *
 *  Created on: Apr 14, 2020
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <stddef.h>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>

#include "MoveType.hpp"

// TODO: Maybe put all moves into a class hierarchy?

namespace netrax {
struct AnnotatedNetwork;
struct Edge;
struct Node;
// The moves correspond to the rNNI moves and rSPR moves from this paper: https://doi.org/10.1371/journal.pcbi.1005611

struct GeneralMove {
    GeneralMove(MoveType type, size_t edge_orig_idx) :
            moveType(type), edge_orig_idx(edge_orig_idx) {
    }
    MoveType moveType;
    size_t edge_orig_idx;

    GeneralMove(GeneralMove&& rhs) : moveType{rhs.moveType}, edge_orig_idx(rhs.edge_orig_idx) {}

    GeneralMove(const GeneralMove& rhs) : moveType{rhs.moveType}, edge_orig_idx(rhs.edge_orig_idx) {}

    GeneralMove& operator =(GeneralMove&& rhs)
    {
        if (this != &rhs)
        {
            moveType = rhs.moveType;
            edge_orig_idx = rhs.edge_orig_idx;
        }
        return *this;
    }

    GeneralMove& operator =(const GeneralMove& rhs)
    {
        if (this != &rhs)
        {
            moveType = rhs.moveType;
            edge_orig_idx = rhs.edge_orig_idx;
        }
        return *this;
    }
};

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

struct ArcInsertionMove: public GeneralMove {
    ArcInsertionMove(size_t edge_orig_idx) :
        GeneralMove(MoveType::ArcInsertionMove, edge_orig_idx) {
    }

    ArcInsertionMove() :
        GeneralMove(MoveType::ArcInsertionMove, 0) {
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
bool checkSanity(AnnotatedNetwork& ann_network, ArcInsertionMove& move);
bool checkSanity(AnnotatedNetwork& ann_network, RNNIMove& move);
bool checkSanity(AnnotatedNetwork& ann_network, RSPRMove& move);
bool checkSanity(AnnotatedNetwork& ann_network, GeneralMove* move);

template <typename T>
bool checkSanity(AnnotatedNetwork& ann_network, std::vector<T>& moves) {
    bool sane = true;
    for (size_t i = 0; i < moves.size(); ++i) {
        sane &= checkSanity(ann_network, moves[i]);
    }
    return sane;
}

std::vector<double> get_edge_lengths(AnnotatedNetwork &ann_network, size_t pmatrix_index);

std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network, const Edge *edge);
std::vector<RNNIMove> possibleRNNIMoves(AnnotatedNetwork &ann_network);
std::vector<RSPRMove> possibleRSPRMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleRSPR1Moves(AnnotatedNetwork &ann_network);

std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleTailMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, const Edge *edge, bool noRSPR1Moves = false);
std::vector<RSPRMove> possibleHeadMoves(AnnotatedNetwork &ann_network, bool noRSPR1Moves = false);

std::vector<GeneralMove*> possibleMoves(AnnotatedNetwork& ann_network, std::vector<MoveType> types);

void performMove(AnnotatedNetwork &ann_network, RNNIMove &move);
void performMove(AnnotatedNetwork &ann_network, RSPRMove &move);
void undoMove(AnnotatedNetwork &ann_network, RNNIMove &move);
void undoMove(AnnotatedNetwork &ann_network, RSPRMove &move);

std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network,
        const Edge *edge, bool noDeltaPlus = false);
std::vector<ArcInsertionMove> possibleArcInsertionMoves(AnnotatedNetwork &ann_network, bool noDeltaPlus = false);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network, Node *v, size_t edge_orig_idx,
        MoveType moveType = MoveType::ArcRemovalMove);
std::vector<ArcRemovalMove> possibleArcRemovalMoves(AnnotatedNetwork &ann_network);

std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network,
        const Edge *edge);
std::vector<ArcInsertionMove> possibleDeltaPlusMoves(AnnotatedNetwork &ann_network);
std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network, Node *v, size_t edge_orig_idx);
std::vector<ArcRemovalMove> possibleDeltaMinusMoves(AnnotatedNetwork &ann_network);

void performMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);
void performMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);
void undoMove(AnnotatedNetwork &ann_network, ArcInsertionMove &move);
void undoMove(AnnotatedNetwork &ann_network, ArcRemovalMove &move);

std::string toString(RNNIMove &move);
std::string toString(RSPRMove &move);
std::string toString(ArcInsertionMove &move);
std::string toString(ArcRemovalMove &move);

std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RNNIMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, RSPRMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, ArcRemovalMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RNNIMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network, RSPRMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcInsertionMove &move);
std::unordered_set<size_t> brlenOptCandidatesUndo(AnnotatedNetwork &ann_network,
        ArcRemovalMove &move);

void performMove(AnnotatedNetwork &ann_network, GeneralMove *move);
void undoMove(AnnotatedNetwork &ann_network, GeneralMove *move);
std::string toString(GeneralMove *move);
std::unordered_set<size_t> brlenOptCandidates(AnnotatedNetwork &ann_network, GeneralMove* move);

ArcInsertionMove randomArcInsertionMove(AnnotatedNetwork &ann_network);
ArcRemovalMove randomArcRemovalMove(AnnotatedNetwork &ann_network);
ArcInsertionMove randomDeltaPlusMove(AnnotatedNetwork &ann_network);
ArcRemovalMove randomDeltaMinusMove(AnnotatedNetwork &ann_network);
RNNIMove randomRNNIMove(AnnotatedNetwork &ann_network);
RSPRMove randomRSPRMove(AnnotatedNetwork &ann_network);
RSPRMove randomRSPR1Move(AnnotatedNetwork &ann_network);
RSPRMove randomTailMove(AnnotatedNetwork &ann_network);
RSPRMove randomHeadMove(AnnotatedNetwork &ann_network);

GeneralMove* copyMove(GeneralMove* move);

}
