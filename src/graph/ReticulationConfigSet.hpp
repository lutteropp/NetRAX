#pragma once

#include <cstddef>
#include <vector>

namespace netrax {

enum class ReticulationState {
  DONT_CARE = 0,
  TAKE_FIRST_PARENT = 1,
  TAKE_SECOND_PARENT = 2,
  INVALID = 3
};

// A ReticulationConfigSet is a big OR, grouping multiple reticulation configs
// that share some characteristic.
struct ReticulationConfigSet {
  std::vector<std::vector<ReticulationState>> configs;
  size_t max_reticulations = 0;

  bool operator==(const ReticulationConfigSet &other) const {
    if (max_reticulations != other.max_reticulations) {
      return false;
    }
    if (configs.size() != other.configs.size()) {
      return false;
    }
    for (size_t i = 0; i < configs.size(); ++i) {
      bool found = false;
      for (size_t j = 0; j < other.configs.size(); ++j) {
        if (configs[i] == other.configs[j]) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }

    return true;
  }

  ReticulationConfigSet() = default;

  ReticulationConfigSet(size_t max_reticulations)
      : max_reticulations(max_reticulations) {}

  ReticulationConfigSet(ReticulationConfigSet &&rhs)
      : max_reticulations{rhs.max_reticulations} {
    configs = std::move(rhs.configs);
  }

  ReticulationConfigSet(const ReticulationConfigSet &rhs)
      : max_reticulations{rhs.max_reticulations} {
    configs.clear();
    for (size_t i = 0; i < rhs.configs.size(); ++i) {
      configs.emplace_back(rhs.configs[i]);
    }
  }

  ReticulationConfigSet &operator=(ReticulationConfigSet &&rhs) {
    if (this != &rhs) {
      max_reticulations = rhs.max_reticulations;
      configs = std::move(rhs.configs);
    }
    return *this;
  }

  ReticulationConfigSet &operator=(const ReticulationConfigSet &rhs) {
    if (this != &rhs) {
      max_reticulations = rhs.max_reticulations;
      configs.clear();
      for (size_t i = 0; i < rhs.configs.size(); ++i) {
        configs.emplace_back(rhs.configs[i]);
      }
    }
    return *this;
  }

  bool empty() const { return configs.empty(); }
};

double computeReticulationConfigProb(
    const ReticulationConfigSet &choices,
    const std::vector<double> &firstParentLogProbs,
    const std::vector<double> &secondParentLogProbs);
double computeReticulationConfigLogProb(
    const ReticulationConfigSet &choices,
    const std::vector<double> &firstParentLogProbs,
    const std::vector<double> &secondParentLogProbs);
bool reticulationConfigsCompatible(const ReticulationConfigSet &left,
                                   const ReticulationConfigSet &right);
void printReticulationChoices(const ReticulationConfigSet &reticulationChoices);
ReticulationConfigSet combineReticulationChoices(
    const ReticulationConfigSet &left, const ReticulationConfigSet &right);
void simplifyReticulationChoices(ReticulationConfigSet &res);
bool validReticulationChoices(const std::vector<ReticulationState> &choices);
void addOrReticulationChoices(ReticulationConfigSet& rcs, const ReticulationConfigSet& moreChoices);

}  // namespace netrax