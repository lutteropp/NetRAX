#pragma once

#include <vector>
#include <limits>

namespace netrax {

struct RSPRData {
    size_t x_prime_clv_index = 0;
    size_t y_prime_clv_index = 0;
    size_t x_clv_index = 0;
    size_t y_clv_index = 0;
    size_t z_clv_index = 0;

    std::vector<double> x_z_len = {0};
    std::vector<double> z_y_len = {0};
    std::vector<double> x_prime_y_prime_len = {0};

    RSPRData() = default;

    RSPRData(RSPRData&& rhs) : x_prime_clv_index{rhs.x_prime_clv_index}, y_prime_clv_index{rhs.y_prime_clv_index}, x_clv_index{rhs.x_clv_index}, y_clv_index{rhs.y_clv_index}, z_clv_index{rhs.z_clv_index}, x_z_len{rhs.x_z_len}, z_y_len{rhs.z_y_len}, x_prime_y_prime_len{rhs.x_prime_y_prime_len} {}

    RSPRData(const RSPRData& rhs) : x_prime_clv_index{rhs.x_prime_clv_index}, y_prime_clv_index{rhs.y_prime_clv_index}, x_clv_index{rhs.x_clv_index}, y_clv_index{rhs.y_clv_index}, z_clv_index{rhs.z_clv_index}, x_z_len{rhs.x_z_len}, z_y_len{rhs.z_y_len}, x_prime_y_prime_len{rhs.x_prime_y_prime_len} {}

    RSPRData& operator =(RSPRData&& rhs) {
        if (this != &rhs) {
            x_prime_clv_index = rhs.x_prime_clv_index;
            y_prime_clv_index = rhs.y_prime_clv_index;
            x_clv_index = rhs.x_clv_index;
            y_clv_index = rhs.y_clv_index;
            z_clv_index = rhs.z_clv_index;
            x_z_len = rhs.x_z_len;
            z_y_len = rhs.z_y_len;
            x_prime_y_prime_len = rhs.x_prime_y_prime_len;
        }
        return *this;
    }

    RSPRData& operator =(const RSPRData& rhs) {
        if (this != &rhs) {
            x_prime_clv_index = rhs.x_prime_clv_index;
            y_prime_clv_index = rhs.y_prime_clv_index;
            x_clv_index = rhs.x_clv_index;
            y_clv_index = rhs.y_clv_index;
            z_clv_index = rhs.z_clv_index;
            x_z_len = rhs.x_z_len;
            z_y_len = rhs.z_y_len;
            x_prime_y_prime_len = rhs.x_prime_y_prime_len;
        }
        return *this;
    }
};

}