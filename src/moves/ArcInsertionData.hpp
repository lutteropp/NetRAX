#pragma once

#include <limits>
#include <vector>

namespace netrax {

struct ArcInsertionData {
  size_t a_clv_index = std::numeric_limits<size_t>::max();
  size_t b_clv_index = std::numeric_limits<size_t>::max();
  size_t c_clv_index = std::numeric_limits<size_t>::max();
  size_t d_clv_index = std::numeric_limits<size_t>::max();

  size_t b_first_parent_clv_index = std::numeric_limits<size_t>::max();
  size_t d_first_parent_clv_index = std::numeric_limits<size_t>::max();

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

  ArcInsertionData() = default;

  ArcInsertionData(ArcInsertionData &&rhs)
      : a_clv_index{rhs.a_clv_index},
        b_clv_index{rhs.b_clv_index},
        c_clv_index{rhs.c_clv_index},
        d_clv_index{rhs.d_clv_index},
        b_first_parent_clv_index{rhs.b_first_parent_clv_index},
        d_first_parent_clv_index{rhs.d_first_parent_clv_index},
        u_v_len{rhs.u_v_len},
        c_v_len{rhs.c_v_len},
        a_u_len{rhs.a_u_len},
        wanted_u_clv_index{rhs.wanted_u_clv_index},
        wanted_v_clv_index{rhs.wanted_v_clv_index},
        wanted_au_pmatrix_index{rhs.wanted_au_pmatrix_index},
        wanted_ub_pmatrix_index{rhs.wanted_ub_pmatrix_index},
        wanted_cv_pmatrix_index{rhs.wanted_cv_pmatrix_index},
        wanted_vd_pmatrix_index{rhs.wanted_vd_pmatrix_index},
        wanted_uv_pmatrix_index{rhs.wanted_uv_pmatrix_index},
        ab_pmatrix_index{rhs.ab_pmatrix_index},
        cd_pmatrix_index{rhs.cd_pmatrix_index},
        a_b_len{rhs.a_b_len},
        c_d_len{rhs.c_d_len},
        v_d_len{rhs.v_d_len},
        u_b_len{rhs.u_b_len} {}

  ArcInsertionData(const ArcInsertionData &rhs)
      : a_clv_index{rhs.a_clv_index},
        b_clv_index{rhs.b_clv_index},
        c_clv_index{rhs.c_clv_index},
        d_clv_index{rhs.d_clv_index},
        b_first_parent_clv_index{rhs.b_first_parent_clv_index},
        d_first_parent_clv_index{rhs.d_first_parent_clv_index},
        u_v_len{rhs.u_v_len},
        c_v_len{rhs.c_v_len},
        a_u_len{rhs.a_u_len},
        wanted_u_clv_index{rhs.wanted_u_clv_index},
        wanted_v_clv_index{rhs.wanted_v_clv_index},
        wanted_au_pmatrix_index{rhs.wanted_au_pmatrix_index},
        wanted_ub_pmatrix_index{rhs.wanted_ub_pmatrix_index},
        wanted_cv_pmatrix_index{rhs.wanted_cv_pmatrix_index},
        wanted_vd_pmatrix_index{rhs.wanted_vd_pmatrix_index},
        wanted_uv_pmatrix_index{rhs.wanted_uv_pmatrix_index},
        ab_pmatrix_index{rhs.ab_pmatrix_index},
        cd_pmatrix_index{rhs.cd_pmatrix_index},
        a_b_len{rhs.a_b_len},
        c_d_len{rhs.c_d_len},
        v_d_len{rhs.v_d_len},
        u_b_len{rhs.u_b_len} {}

  ArcInsertionData &operator=(ArcInsertionData &&rhs) {
    if (this != &rhs) {
      a_clv_index = rhs.a_clv_index;
      b_clv_index = rhs.b_clv_index;
      c_clv_index = rhs.c_clv_index;
      d_clv_index = rhs.d_clv_index;
      b_first_parent_clv_index = rhs.b_first_parent_clv_index;
      d_first_parent_clv_index = rhs.d_first_parent_clv_index;
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
    return *this;
  }

  ArcInsertionData &operator=(const ArcInsertionData &rhs) {
    if (this != &rhs) {
      a_clv_index = rhs.a_clv_index;
      b_clv_index = rhs.b_clv_index;
      c_clv_index = rhs.c_clv_index;
      d_clv_index = rhs.d_clv_index;
      b_first_parent_clv_index = rhs.b_first_parent_clv_index;
      d_first_parent_clv_index = rhs.d_first_parent_clv_index;
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
    return *this;
  }

  bool operator==(const ArcInsertionData &rhs) const {
    return ((this->a_clv_index == rhs.a_clv_index) &&
            (this->b_clv_index == rhs.b_clv_index) &&
            (this->c_clv_index == rhs.c_clv_index) &&
            (this->d_clv_index == rhs.d_clv_index) &&
            (this->b_first_parent_clv_index == rhs.b_first_parent_clv_index) &&
            (this->d_first_parent_clv_index == rhs.d_first_parent_clv_index)
            /*&& (this->u_v_len == rhs.u_v_len)
        && (this->c_v_len == rhs.c_v_len)
        && (this->a_u_len == rhs.a_u_len)*/
            && (this->wanted_u_clv_index == rhs.wanted_u_clv_index) &&
            (this->wanted_v_clv_index == rhs.wanted_v_clv_index) &&
            (this->wanted_au_pmatrix_index == rhs.wanted_au_pmatrix_index) &&
            (this->wanted_ub_pmatrix_index == rhs.wanted_ub_pmatrix_index) &&
            (this->wanted_cv_pmatrix_index == rhs.wanted_cv_pmatrix_index) &&
            (this->wanted_vd_pmatrix_index == rhs.wanted_vd_pmatrix_index) &&
            (this->wanted_uv_pmatrix_index == rhs.wanted_uv_pmatrix_index) &&
            (this->ab_pmatrix_index == rhs.ab_pmatrix_index) &&
            (this->cd_pmatrix_index == rhs.cd_pmatrix_index)
            /*&& (this->a_b_len == rhs.a_b_len)
        && (this->c_d_len == rhs.c_d_len)
        && (this->v_d_len == rhs.v_d_len)
        && (this->u_b_len == rhs.u_b_len)*/
    );
  }
};

}  // namespace netrax