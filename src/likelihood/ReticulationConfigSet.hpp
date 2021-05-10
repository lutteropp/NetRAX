#pragma once

#include <vector>

namespace netrax {

enum class ReticulationState {
    DONT_CARE = 0,
    TAKE_FIRST_PARENT = 1,
    TAKE_SECOND_PARENT = 2,
    INVALID = 3
};

struct ReticulationConfigSet {
    std::vector<std::vector<ReticulationState> > configs;
    size_t max_reticulations = 0;

    bool operator==(const ReticulationConfigSet& other) const {
        if (max_reticulations != other.max_reticulations) {
            return false;
        }
        if (configs.size() != other.configs.size()) {
            return false;
        }
        for (size_t i = 0; i < configs.size(); ++i) {
            if (configs[i].size() != other.configs[i].size()) {
                return false;
            }
            for (size_t j = 0; j < configs[i].size(); ++j) {
                if (configs[i][j] != other.configs[i][j]) {
                    return false;
                }
            }
        }

        return true;
    }

    ReticulationConfigSet() = default;

    ReticulationConfigSet(size_t max_reticulations) : max_reticulations(max_reticulations) {}

    ReticulationConfigSet(ReticulationConfigSet&& rhs) : max_reticulations{rhs.max_reticulations}
    {
        configs = std::move(rhs.configs);
    }

    ReticulationConfigSet(const ReticulationConfigSet& rhs)
      : max_reticulations{rhs.max_reticulations}
    {
        configs.clear();
        for (size_t i = 0; i < rhs.configs.size(); ++i) {
            configs.emplace_back(rhs.configs[i]);
        }
    }

    ReticulationConfigSet& operator =(ReticulationConfigSet&& rhs)
    {
        if (this != &rhs)
        {
            max_reticulations = rhs.max_reticulations;
            configs = std::move(rhs.configs);
        }
        return *this;
    }

    ReticulationConfigSet& operator =(const ReticulationConfigSet& rhs)
    {
        if (this != &rhs)
        {
            max_reticulations = rhs.max_reticulations;
            configs.clear();
            for (size_t i = 0; i < rhs.configs.size(); ++i) {
                configs.emplace_back(rhs.configs[i]);
            }
        }
        return *this;
    }

    bool empty() const {
        return configs.empty();
    }
};

}