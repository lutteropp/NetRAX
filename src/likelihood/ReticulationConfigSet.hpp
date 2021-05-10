#pragma once

#include <vector>
#include <cassert>
#include <iostream>

namespace netrax {

enum class ReticulationState {
    DONT_CARE = 0,
    TAKE_FIRST_PARENT = 1,
    TAKE_SECOND_PARENT = 2,
    INVALID = 3
};

inline ReticulationState andState(const ReticulationState& a, const ReticulationState& b) {
    if (a == b) {
        return a;
    }
    if (a == ReticulationState::DONT_CARE) {
        return b;
    }
    if (b == ReticulationState::DONT_CARE) {
        return a;
    }
    return ReticulationState::INVALID;
}

inline ReticulationState orState(const ReticulationState& a, const ReticulationState& b) {
    if (a == b) {
        return a;
    }
    if (a == ReticulationState::DONT_CARE || b == ReticulationState::DONT_CARE) {
        return ReticulationState::DONT_CARE;
    }
    if (a == ReticulationState::INVALID || b == ReticulationState::INVALID) {
        return ReticulationState::INVALID;
    }
    return ReticulationState::DONT_CARE;
}

inline ReticulationState withoutState(const ReticulationState& a, const ReticulationState& b) {
    if (a == ReticulationState::DONT_CARE && b == ReticulationState::DONT_CARE) {
        return ReticulationState::DONT_CARE;
    }
    if (a == ReticulationState::DONT_CARE && b == ReticulationState::TAKE_FIRST_PARENT) {
        return ReticulationState::TAKE_SECOND_PARENT;
    }
    if (a == ReticulationState::DONT_CARE && b == ReticulationState::TAKE_SECOND_PARENT) {
        return ReticulationState::TAKE_FIRST_PARENT;
    }
    if (b == ReticulationState::DONT_CARE && a == ReticulationState::TAKE_FIRST_PARENT) {
        return ReticulationState::TAKE_SECOND_PARENT;
    }
    if (b == ReticulationState::DONT_CARE && a == ReticulationState::TAKE_SECOND_PARENT) {
        return ReticulationState::TAKE_FIRST_PARENT;
    }
    return ReticulationState::INVALID;
}

struct ReticulationConfig {
    std::vector<ReticulationState> config;

    ReticulationConfig(size_t max_reticulations) {
        config = std::vector<ReticulationState>(max_reticulations, ReticulationState::DONT_CARE);
    }

    ReticulationConfig(ReticulationConfig&& rhs) {
        config = std::move(rhs.config);
    }

    ReticulationConfig(const ReticulationConfig& rhs)
    {
        config = rhs.config;
    }

    ReticulationConfig& operator =(ReticulationConfig&& rhs) {
        if (this != &rhs)
        {
            config = std::move(rhs.config);
        }
        return *this;
    }

    ReticulationConfig& operator =(const ReticulationConfig& rhs) {
        if (this != &rhs)
        {
            config = rhs.config;
        }
        return *this;
    }

    ReticulationState& operator [](size_t idx) {
        return config[idx];
    }

    const ReticulationState& operator [](size_t idx) const {
        return config[idx];
    }

    size_t size() const {
        return config.size();
    }

    bool operator==(const ReticulationConfig& other) const {
        return config == other.config;
    }

    bool operator!=(const ReticulationConfig& other) const {
        return config != other.config;
    }

    bool valid() const {
        for (size_t i = 0; i < config.size(); ++i) {
            if (config[i] == ReticulationState::INVALID) {
                return false;
            }
        }
        return true;
    }

    bool trivial() const {
        for (size_t i = 0; i < config.size(); ++i) {
            if (config[i] != ReticulationState::DONT_CARE) {
                return false;
            }
        }
        return true;
    }

    ReticulationConfig operator &(const ReticulationConfig& rhs) const {
        assert(config.size() == rhs.config.size());
        ReticulationConfig res(config.size());
        for (size_t i = 0; i < config.size(); ++i) {
            res.config[i] = andState(config[i], rhs.config[i]);
        }
        return res;
    }

    ReticulationConfig operator |(const ReticulationConfig& rhs) const {
        assert(config.size() == rhs.config.size());
        ReticulationConfig res(config.size());
        for (size_t i = 0; i < config.size(); ++i) {
            res.config[i] = orState(config[i], rhs.config[i]);
        }
        return res;
    }

    ReticulationConfig operator /(const ReticulationConfig& rhs) const {
        assert(config.size() == rhs.config.size());
        ReticulationConfig res(config.size());
        for (size_t i = 0; i < config.size(); ++i) {
            res.config[i] = withoutState(config[i], rhs.config[i]);
        }
        return res;
    }
};

void printReticulationChoices(const ReticulationConfig& reticulationChoices);

struct ReticulationConfigSet {
    std::vector<ReticulationConfig> configs;
    size_t max_reticulations = 0;

    bool operator==(const ReticulationConfigSet& other) const {
        if (max_reticulations != other.max_reticulations) {
            return false;
        }
        if (configs.size() != other.configs.size()) {
            return false;
        }
        for (size_t i = 0; i < configs.size(); ++i) {
            if (configs[i] != other.configs[i]) {
                return false;
            }
        }

        return true;
    }

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

    bool valid() const {
        for (size_t i = 0; i < configs.size(); ++i) {
            if (!configs[i].valid()) {
                return false;
            }
        }
        return true;
    }

    bool trivial() const {
        for (size_t i = 0; i < configs.size(); ++i) {
            if (!configs[i].trivial()) {
                return false;
            }
        }
        return true;
    }

    void clear() {
        configs.clear();
    }

    void simplifyReticulationChoices();

    ReticulationConfigSet operator &(const ReticulationConfigSet& rhs) const {
        if (empty() || rhs.empty()) {
            return ReticulationConfigSet(max_reticulations);
        }

        assert(max_reticulations == rhs.max_reticulations);
        ReticulationConfigSet res(max_reticulations);
        for (size_t i = 0; i < configs.size(); ++i) {
            for (size_t j = 0; j < rhs.configs.size(); ++j) {
                ReticulationConfig combined = (configs[i] & rhs.configs[j]);
                if (combined.valid()) {
                    res.configs.emplace_back(combined);
                }
            }
        }
        res.simplifyReticulationChoices();
        return res;
    }

    ReticulationConfigSet operator |(const ReticulationConfigSet& rhs) const {
        assert(max_reticulations == rhs.max_reticulations);
        ReticulationConfigSet res(max_reticulations);
        for (size_t i = 0; i < configs.size(); ++i) {
            res.configs.emplace_back(configs[i]);
        }
        for (size_t i = 0; i < rhs.configs.size(); ++i) {
            res.configs.emplace_back(rhs.configs[i]);
        }
        res.simplifyReticulationChoices();
        return res;
    }

    bool hasConfig(const ReticulationConfig& conf) const {
        for (size_t i = 0; i < configs.size(); ++i) {
            if (configs[i] == conf) {
                return true;
            }
        }
        return false;
    }

    ReticulationConfigSet operator /(const ReticulationConfigSet& rhs) const {
        assert(max_reticulations == rhs.max_reticulations);
        ReticulationConfigSet res(max_reticulations);

        for (size_t i = 0; i < configs.size(); ++i) {
            for (size_t j = 0; j < rhs.configs.size(); ++j) {
                ReticulationConfig without = (configs[i] / rhs.configs[j]);
                if (without.valid() && !hasConfig(without)) {
                    res.configs.emplace_back(without);
                }
            }
        }
        return res;
    }
};

//double computeReticulationChoicesLogProb(const std::vector<ReticulationState>& choices, const std::vector<double>& reticulationProbs);

//bool reticulationChoicesCompatible(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
bool reticulationConfigsCompatible(const ReticulationConfigSet& left, const ReticulationConfigSet& right);

void printReticulationChoices(const ReticulationConfigSet& reticulationChoices);

}