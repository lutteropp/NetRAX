#pragma once

#include <vector>
#include <limits>

namespace netrax {

enum class RNNIMoveType {
    ONE, ONE_STAR, TWO, TWO_STAR, THREE, THREE_STAR, FOUR
};

struct RNNIData {
    size_t u_clv_index = 0;
    size_t v_clv_index = 0;
    size_t s_clv_index = 0;
    size_t t_clv_index = 0;
    RNNIMoveType type = RNNIMoveType::ONE;

    RNNIData() = default;

    RNNIData(RNNIData&& rhs) : u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, s_clv_index{rhs.s_clv_index}, t_clv_index{rhs.t_clv_index}, type{rhs.type} {}

    RNNIData(const RNNIData& rhs) : u_clv_index{rhs.u_clv_index}, v_clv_index{rhs.v_clv_index}, s_clv_index{rhs.s_clv_index}, t_clv_index{rhs.t_clv_index}, type{rhs.type} {}

    RNNIData& operator =(RNNIData&& rhs) {
        if (this != &rhs) {
            u_clv_index = rhs.u_clv_index;
            v_clv_index = rhs.v_clv_index;
            s_clv_index = rhs.s_clv_index;
            t_clv_index = rhs.t_clv_index;
            type = rhs.type;
        }
        return *this;
    }

    RNNIData& operator =(const RNNIData& rhs) {
        if (this != &rhs) {
            u_clv_index = rhs.u_clv_index;
            v_clv_index = rhs.v_clv_index;
            s_clv_index = rhs.s_clv_index;
            t_clv_index = rhs.t_clv_index;
            type = rhs.type;
        }
        return *this;
    }
};

inline bool operator==(const RNNIData& lhs, const RNNIData& rhs){ 
    return(
        (lhs.u_clv_index == rhs.u_clv_index)
        && (lhs.v_clv_index == rhs.v_clv_index)
        && (lhs.s_clv_index == rhs.s_clv_index)
        && (lhs.t_clv_index == rhs.t_clv_index)
        && (lhs.type == rhs.type)
    );
}

inline bool operator!=(const RNNIData& lhs, const RNNIData& rhs){ return !(lhs == rhs); }

}