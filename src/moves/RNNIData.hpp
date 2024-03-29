#pragma once

#include <limits>
#include <vector>

namespace netrax {

enum class RNNIMoveType {
  ONE,
  ONE_STAR,
  TWO,
  TWO_STAR,
  THREE,
  THREE_STAR,
  FOUR
};

struct RNNIData {
  size_t u_clv_index = std::numeric_limits<size_t>::max();
  size_t v_clv_index = std::numeric_limits<size_t>::max();
  size_t s_clv_index = std::numeric_limits<size_t>::max();
  size_t t_clv_index = std::numeric_limits<size_t>::max();
  size_t u_first_parent_clv_index = std::numeric_limits<size_t>::max();
  size_t v_first_parent_clv_index = std::numeric_limits<size_t>::max();
  size_t s_first_parent_clv_index = std::numeric_limits<size_t>::max();
  size_t t_first_parent_clv_index = std::numeric_limits<size_t>::max();
  RNNIMoveType type = RNNIMoveType::ONE;

  RNNIData() = default;

  RNNIData(RNNIData &&rhs)
      : u_clv_index{rhs.u_clv_index},
        v_clv_index{rhs.v_clv_index},
        s_clv_index{rhs.s_clv_index},
        t_clv_index{rhs.t_clv_index},
        u_first_parent_clv_index{rhs.u_first_parent_clv_index},
        v_first_parent_clv_index{rhs.v_first_parent_clv_index},
        s_first_parent_clv_index{rhs.s_first_parent_clv_index},
        t_first_parent_clv_index{rhs.t_first_parent_clv_index},
        type{rhs.type} {}

  RNNIData(const RNNIData &rhs)
      : u_clv_index{rhs.u_clv_index},
        v_clv_index{rhs.v_clv_index},
        s_clv_index{rhs.s_clv_index},
        t_clv_index{rhs.t_clv_index},
        u_first_parent_clv_index{rhs.u_first_parent_clv_index},
        v_first_parent_clv_index{rhs.v_first_parent_clv_index},
        s_first_parent_clv_index{rhs.s_first_parent_clv_index},
        t_first_parent_clv_index{rhs.t_first_parent_clv_index},
        type{rhs.type} {}

  RNNIData &operator=(RNNIData &&rhs) {
    if (this != &rhs) {
      u_clv_index = rhs.u_clv_index;
      v_clv_index = rhs.v_clv_index;
      s_clv_index = rhs.s_clv_index;
      t_clv_index = rhs.t_clv_index;
      u_first_parent_clv_index = rhs.u_first_parent_clv_index;
      v_first_parent_clv_index = rhs.v_first_parent_clv_index;
      s_first_parent_clv_index = rhs.s_first_parent_clv_index;
      t_first_parent_clv_index = rhs.t_first_parent_clv_index;
      type = rhs.type;
    }
    return *this;
  }

  RNNIData &operator=(const RNNIData &rhs) {
    if (this != &rhs) {
      u_clv_index = rhs.u_clv_index;
      v_clv_index = rhs.v_clv_index;
      s_clv_index = rhs.s_clv_index;
      t_clv_index = rhs.t_clv_index;
      u_first_parent_clv_index = rhs.u_first_parent_clv_index;
      v_first_parent_clv_index = rhs.v_first_parent_clv_index;
      s_first_parent_clv_index = rhs.s_first_parent_clv_index;
      t_first_parent_clv_index = rhs.t_first_parent_clv_index;
      type = rhs.type;
    }
    return *this;
  }

  bool operator==(const RNNIData &rhs) const {
    return ((this->u_clv_index == rhs.u_clv_index) &&
            (this->v_clv_index == rhs.v_clv_index) &&
            (this->s_clv_index == rhs.s_clv_index) &&
            (this->t_clv_index == rhs.t_clv_index) &&
            (this->u_first_parent_clv_index == rhs.u_first_parent_clv_index) &&
            (this->v_first_parent_clv_index == rhs.v_first_parent_clv_index) &&
            (this->s_first_parent_clv_index == rhs.s_first_parent_clv_index) &&
            (this->t_first_parent_clv_index == rhs.t_first_parent_clv_index) &&
            (this->type == rhs.type));
  }
};

}  // namespace netrax