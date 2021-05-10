#include "ReticulationConfigSet.hpp"

#include <cassert>
#include <iostream>
#include "mpreal.h"

namespace netrax {
 
bool reticulationChoicesCompatible(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right) {
    assert(left.size() == right.size());
    for (size_t i = 0; i < left.size(); ++i) {
        if (left[i] != ReticulationState::DONT_CARE) {
            if ((right[i] != ReticulationState::DONT_CARE) && (right[i] != left[i])) {
                return false;
            }
        }
        if ((left[i] == ReticulationState::INVALID) || (right[i] == ReticulationState::INVALID)) {
            return false;
        }
    }
    return true;
}

std::vector<ReticulationState> combineReticulationChoices(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right) {
    assert(reticulationChoicesCompatible(left, right));
    assert(left.size() == right.size());
    std::vector<ReticulationState> res = left;
    for (size_t i = 0; i < res.size(); ++i) {
        if (left[i] == ReticulationState::DONT_CARE) {
            res[i] = right[i];
        }
    }
    return res;
}

//bool reticulationChoicesCompatible(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
bool reticulationConfigsCompatible(const ReticulationConfigSet& left, const ReticulationConfigSet& right) {
    return !((left & right).empty());
}

void ReticulationConfigSet::simplifyReticulationChoices() {
    // simplify the reticulation choices. E.g., if we have both 00 and 01, summarize them to 0-
    bool shrinked = true;
    while (shrinked) {
        shrinked = false;

        // fist: search for duplicates
        for (size_t i = 0; i < configs.size(); ++i) {
            for (size_t j = i+1; j < configs.size(); ++j) {
                if (configs[i] == configs[j]) {
                    // remove res.configs[j]
                    std::swap(configs[j], configs[configs.size() - 1]);
                    configs.pop_back();
                    shrinked = true;
                    break;
                }
            }
            if (shrinked) {
                break;
            }
        }

        if (shrinked) {
            continue;
        }

        for (size_t i = 0; i < configs.size(); ++i) {
            for (size_t j = 0; j < max_reticulations; ++j) {
                if (configs[i][j] != ReticulationState::DONT_CARE) {
                    ReticulationConfig query = configs[i];
                    if (configs[i][j] == ReticulationState::TAKE_FIRST_PARENT) {
                        query[j] = ReticulationState::TAKE_SECOND_PARENT;
                    } else {
                        query[j] = ReticulationState::TAKE_FIRST_PARENT;
                    }
                    // search if the query is present in res
                    for (size_t k = 0; k < configs.size(); ++k) {
                        if (k == i) {
                            continue;
                        }
                        if (configs[k] == query) {
                            configs[i][j] = ReticulationState::DONT_CARE;
                            // remove res.configs[k]
                            std::swap(configs[k], configs[configs.size() - 1]);
                            configs.pop_back();
                            shrinked = true;
                            break;
                        }
                        if (configs[k] == configs[i]) {
                            // remove res.configs[k]
                            std::swap(configs[k], configs[configs.size() - 1]);
                            configs.pop_back();
                            shrinked = true;
                            break;
                        }
                    }
                }
                if (shrinked) {
                    break;
                }
            }
            if (shrinked) {
                break;
            }
        }
    }
}

void printReticulationChoices(const ReticulationConfig& reticulationChoices) {
    for (size_t i = 0; i < reticulationChoices.size(); ++i) {
        char c;
        if (reticulationChoices[i] == ReticulationState::TAKE_FIRST_PARENT) {
            c = '0';
        } else if (reticulationChoices[i] == ReticulationState::TAKE_SECOND_PARENT) {
            c = '1';
        } else {
            c = '-';
        }
        std::cout << c;
    }
    std::cout << "\n";
}

void printReticulationChoices(const ReticulationConfigSet& reticulationChoices) {
    for (size_t i = 0; i < reticulationChoices.configs.size(); ++i) {
        printReticulationChoices(reticulationChoices.configs[i]);
    }
}

}