#pragma once

#include <vector>
#include <limits>

namespace netrax {

struct ArcRemovalData {
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

    std::vector<std::pair<size_t, size_t> > remapped_clv_indices;
    std::vector<std::pair<size_t, size_t> > remapped_pmatrix_indices;

    ArcRemovalData() = default;

    ArcRemovalData(ArcRemovalData&& rhs) : a_clv_index{rhs.a_clv_index}, b_clv_index{rhs.b_clv_index}, c_clv_index{rhs.c_clv_index}, d_clv_index{rhs.d_clv_index}, u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, u_v_len{rhs.u_v_len}, c_v_len{rhs.c_v_len}, a_u_len{rhs.a_u_len}, au_pmatrix_index{rhs.au_pmatrix_index}, ub_pmatrix_index{rhs.ub_pmatrix_index}, cv_pmatrix_index{rhs.cv_pmatrix_index}, vd_pmatrix_index{rhs.vd_pmatrix_index}, uv_pmatrix_index{rhs.uv_pmatrix_index}, wanted_ab_pmatrix_index{rhs.wanted_ab_pmatrix_index}, wanted_cd_pmatrix_index{rhs.wanted_cd_pmatrix_index}, a_b_len{rhs.a_b_len}, c_d_len{rhs.c_d_len}, v_d_len{rhs.v_d_len}, u_b_len{rhs.u_b_len}, remapped_clv_indices{rhs.remapped_clv_indices}, remapped_pmatrix_indices{rhs.remapped_pmatrix_indices} {}

    ArcRemovalData(const ArcRemovalData& rhs) : a_clv_index{rhs.a_clv_index}, b_clv_index{rhs.b_clv_index}, c_clv_index{rhs.c_clv_index}, d_clv_index{rhs.d_clv_index}, u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, u_v_len{rhs.u_v_len}, c_v_len{rhs.c_v_len}, a_u_len{rhs.a_u_len}, au_pmatrix_index{rhs.au_pmatrix_index}, ub_pmatrix_index{rhs.ub_pmatrix_index}, cv_pmatrix_index{rhs.cv_pmatrix_index}, vd_pmatrix_index{rhs.vd_pmatrix_index}, uv_pmatrix_index{rhs.uv_pmatrix_index}, wanted_ab_pmatrix_index{rhs.wanted_ab_pmatrix_index}, wanted_cd_pmatrix_index{rhs.wanted_cd_pmatrix_index}, a_b_len{rhs.a_b_len}, c_d_len{rhs.c_d_len}, v_d_len{rhs.v_d_len}, u_b_len{rhs.u_b_len}, remapped_clv_indices{rhs.remapped_clv_indices}, remapped_pmatrix_indices{rhs.remapped_pmatrix_indices} {}

    ArcRemovalData& operator =(ArcRemovalData&& rhs) {
        if (this != &rhs) {
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

            remapped_clv_indices = rhs.remapped_clv_indices;
            remapped_pmatrix_indices = rhs.remapped_pmatrix_indices;
        }
        return *this;
    }

    ArcRemovalData& operator =(const ArcRemovalData& rhs) {
        if (this != &rhs) {
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

            remapped_clv_indices = rhs.remapped_clv_indices;
            remapped_pmatrix_indices = rhs.remapped_pmatrix_indices;
        }
        return *this;
    }
};

}