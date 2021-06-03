#pragma once

#include <limits>
#include <vector>

namespace netrax {

struct RSPRData {
  size_t x_prime_clv_index = std::numeric_limits<size_t>::max();
  size_t y_prime_clv_index = std::numeric_limits<size_t>::max();
  size_t x_clv_index = std::numeric_limits<size_t>::max();
  size_t y_clv_index = std::numeric_limits<size_t>::max();
  size_t z_clv_index = std::numeric_limits<size_t>::max();
  size_t y_prime_first_parent_clv_index = std::numeric_limits<size_t>::max();
  size_t y_first_parent_clv_index = std::numeric_limits<size_t>::max();
  size_t z_first_parent_clv_index = std::numeric_limits<size_t>::max();

  std::vector<double> x_z_len = {0};
  std::vector<double> z_y_len = {0};
  std::vector<double> x_prime_y_prime_len = {0};

  RSPRData() = default;

  RSPRData(RSPRData &&rhs)
      : x_prime_clv_index{rhs.x_prime_clv_index},
        y_prime_clv_index{rhs.y_prime_clv_index},
        x_clv_index{rhs.x_clv_index},
        y_clv_index{rhs.y_clv_index},
        z_clv_index{rhs.z_clv_index},
        y_prime_first_parent_clv_index{rhs.y_prime_first_parent_clv_index},
        y_first_parent_clv_index{rhs.y_first_parent_clv_index},
        z_first_parent_clv_index{rhs.z_first_parent_clv_index},
        x_z_len{rhs.x_z_len},
        z_y_len{rhs.z_y_len},
        x_prime_y_prime_len{rhs.x_prime_y_prime_len} {}

  RSPRData(const RSPRData &rhs)
      : x_prime_clv_index{rhs.x_prime_clv_index},
        y_prime_clv_index{rhs.y_prime_clv_index},
        x_clv_index{rhs.x_clv_index},
        y_clv_index{rhs.y_clv_index},
        z_clv_index{rhs.z_clv_index},
        y_prime_first_parent_clv_index{rhs.y_prime_first_parent_clv_index},
        y_first_parent_clv_index{rhs.y_first_parent_clv_index},
        z_first_parent_clv_index{rhs.z_first_parent_clv_index},
        x_z_len{rhs.x_z_len},
        z_y_len{rhs.z_y_len},
        x_prime_y_prime_len{rhs.x_prime_y_prime_len} {}

  RSPRData &operator=(RSPRData &&rhs) {
    if (this != &rhs) {
      x_prime_clv_index = rhs.x_prime_clv_index;
      y_prime_clv_index = rhs.y_prime_clv_index;
      x_clv_index = rhs.x_clv_index;
      y_clv_index = rhs.y_clv_index;
      z_clv_index = rhs.z_clv_index;
      y_prime_first_parent_clv_index = rhs.y_prime_first_parent_clv_index;
      y_first_parent_clv_index = rhs.y_first_parent_clv_index;
      z_first_parent_clv_index = rhs.z_first_parent_clv_index;
      x_z_len = rhs.x_z_len;
      z_y_len = rhs.z_y_len;
      x_prime_y_prime_len = rhs.x_prime_y_prime_len;
    }
    return *this;
  }

  RSPRData &operator=(const RSPRData &rhs) {
    if (this != &rhs) {
      x_prime_clv_index = rhs.x_prime_clv_index;
      y_prime_clv_index = rhs.y_prime_clv_index;
      x_clv_index = rhs.x_clv_index;
      y_clv_index = rhs.y_clv_index;
      z_clv_index = rhs.z_clv_index;
      y_prime_first_parent_clv_index = rhs.y_prime_first_parent_clv_index;
      y_first_parent_clv_index = rhs.y_first_parent_clv_index;
      z_first_parent_clv_index = rhs.z_first_parent_clv_index;
      x_z_len = rhs.x_z_len;
      z_y_len = rhs.z_y_len;
      x_prime_y_prime_len = rhs.x_prime_y_prime_len;
    }
    return *this;
  }

  bool operator==(const RSPRData &rhs) const {
    return ((this->x_prime_clv_index == rhs.x_prime_clv_index) &&
            (this->y_prime_clv_index == rhs.y_prime_clv_index) &&
            (this->x_clv_index == rhs.x_clv_index) &&
            (this->y_clv_index == rhs.y_clv_index) &&
            (this->z_clv_index == rhs.z_clv_index) &&
            (this->y_prime_first_parent_clv_index ==
             rhs.y_prime_first_parent_clv_index) &&
            (this->y_first_parent_clv_index == rhs.y_first_parent_clv_index) &&
            (this->z_first_parent_clv_index == rhs.z_first_parent_clv_index)
            /*&& (this->x_z_len == rhs.x_z_len)
        && (this->z_y_len == rhs.z_y_len)*/
            && (this->x_prime_y_prime_len == rhs.x_prime_y_prime_len));
  }
};

}  // namespace netrax