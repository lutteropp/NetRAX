#include "ReticulationConfigSet.hpp"

#include <cassert>
#include <iostream>
#include "../likelihood/mpreal.h"

namespace netrax {

bool reticulationChoicesCompatible(
    const std::vector<ReticulationState> &left,
    const std::vector<ReticulationState> &right) {
  assert(left.size() == right.size());
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != ReticulationState::DONT_CARE) {
      if ((right[i] != ReticulationState::DONT_CARE) && (right[i] != left[i])) {
        return false;
      }
    }
    if ((left[i] == ReticulationState::INVALID) ||
        (right[i] == ReticulationState::INVALID)) {
      return false;
    }
  }
  return true;
}

void printReticulationChoices(
    const std::vector<ReticulationState> &reticulationChoices) {
  for (size_t i = 0; i < reticulationChoices.size(); ++i) {
    char c;
    if (reticulationChoices[i] == ReticulationState::TAKE_FIRST_PARENT) {
      c = '0';
    } else if (reticulationChoices[i] ==
               ReticulationState::TAKE_SECOND_PARENT) {
      c = '1';
    } else {
      c = '-';
    }
    std::cout << c;
  }
  std::cout << "\n";
}

std::vector<ReticulationState> combineReticulationChoices(
    const std::vector<ReticulationState> &left,
    const std::vector<ReticulationState> &right) {
  assert(left.size() == right.size());
  std::vector<ReticulationState> res = left;
  for (size_t i = 0; i < res.size(); ++i) {
    if (left[i] == ReticulationState::DONT_CARE) {
      res[i] = right[i];
    } else if (right[i] == ReticulationState::DONT_CARE) {
      res[i] = left[i];
    } else if (left[i] != right[i]) {
      res[i] = ReticulationState::INVALID;
    }
  }
  return res;
}

bool validReticulationChoices(const std::vector<ReticulationState> &choices) {
  for (size_t i = 0; i < choices.size(); ++i) {
    if (choices[i] == ReticulationState::INVALID) {
      return false;
    }
  }
  return true;
}

double computeReticulationChoicesLogProb_internal(
    const std::vector<ReticulationState> &choices,
    const std::vector<double> &firstParentLogProbs,
    const std::vector<double> &secondParentLogProbs) {
  double logProb = 0;
  for (size_t i = 0; i < choices.size(); ++i) {
    if (choices[i] != ReticulationState::DONT_CARE) {
      if (choices[i] == ReticulationState::TAKE_FIRST_PARENT) {
        logProb += firstParentLogProbs[i];
      } else {
        logProb += secondParentLogProbs[i];
      }
    }
  }
  if (choices[0][0] != ReticulationState::DONT_CARE) {
    assert(logProb != 0.0);
  }
  return logProb;
}

double computeReticulationChoicesLogProb(
    const std::vector<ReticulationState> &choices,
    const std::vector<double> &firstParentLogProbs,
    const std::vector<double> &secondParentLogProbs) {
  return computeReticulationChoicesLogProb_internal(
      choices, firstParentLogProbs, secondParentLogProbs);
}

double computeReticulationConfigLogProb(
    const ReticulationConfigSet &choices,
    const std::vector<double> &firstParentLogProbs,
    const std::vector<double> &secondParentLogProbs) {
  if (choices.configs.size() == 1) {
    return computeReticulationChoicesLogProb_internal(
        choices.configs[0], firstParentLogProbs, secondParentLogProbs);
  } else {
    mpfr::mpreal prob = 0.0;
    for (size_t i = 0; i < choices.configs.size(); ++i) {
      prob += mpfr::exp(computeReticulationChoicesLogProb_internal(
          choices.configs[i], firstParentLogProbs, secondParentLogProbs));
    }
    return mpfr::log(prob).toDouble();
  }
}

double computeReticulationConfigProb(
    const ReticulationConfigSet &choices,
    const std::vector<double> &firstParentLogProbs,
    const std::vector<double> &secondParentLogProbs) {
  if (choices.configs.size() == 1) {
    return exp(computeReticulationChoicesLogProb_internal(
        choices.configs[0], firstParentLogProbs, secondParentLogProbs));
  } else {
    mpfr::mpreal prob = 0.0;
    for (size_t i = 0; i < choices.configs.size(); ++i) {
      prob += mpfr::exp(computeReticulationChoicesLogProb_internal(
          choices.configs[i], firstParentLogProbs, secondParentLogProbs));
    }
    return prob.toDouble();
  }
}

bool reticulationConfigsCompatible(const ReticulationConfigSet &left,
                                   const ReticulationConfigSet &right) {
  for (size_t i = 0; i < left.configs.size(); ++i) {
    for (size_t j = 0; j < right.configs.size(); ++j) {
      if (reticulationChoicesCompatible(left.configs[i], right.configs[j])) {
        return true;
      }
    }
  }
  return false;
}

void printReticulationChoices(
    const ReticulationConfigSet &reticulationChoices) {
  for (size_t i = 0; i < reticulationChoices.configs.size(); ++i) {
    printReticulationChoices(reticulationChoices.configs[i]);
  }
}

void simplifyReticulationChoices(ReticulationConfigSet &res) {
  // simplify the reticulation choices. E.g., if we have both 00 and 01,
  // summarize them to 0-
  bool shrinked = true;
  while (shrinked) {
    shrinked = false;

    // first: search for duplicates
    for (size_t i = 0; i < res.configs.size(); ++i) {
      for (size_t j = i + 1; j < res.configs.size(); ++j) {
        if (res.configs[i] == res.configs[j]) {
          // remove res.configs[j]
          std::swap(res.configs[j], res.configs[res.configs.size() - 1]);
          res.configs.pop_back();
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

    for (size_t i = 0; i < res.configs.size(); ++i) {
      for (size_t j = 0; j < res.max_reticulations; ++j) {
        if (res.configs[i][j] != ReticulationState::DONT_CARE) {
          std::vector<ReticulationState> query = res.configs[i];
          if (res.configs[i][j] == ReticulationState::TAKE_FIRST_PARENT) {
            query[j] = ReticulationState::TAKE_SECOND_PARENT;
          } else {
            query[j] = ReticulationState::TAKE_FIRST_PARENT;
          }
          // search if the query is present in res
          for (size_t k = 0; k < res.configs.size(); ++k) {
            if (k == i) {
              continue;
            }
            if (res.configs[k] == query) {
              res.configs[i][j] = ReticulationState::DONT_CARE;
              // remove res.configs[k]
              std::swap(res.configs[k], res.configs[res.configs.size() - 1]);
              res.configs.pop_back();
              shrinked = true;
              break;
            }
            if (res.configs[k] == res.configs[i]) {
              // remove res.configs[k]
              std::swap(res.configs[k], res.configs[res.configs.size() - 1]);
              res.configs.pop_back();
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

ReticulationConfigSet combineReticulationChoices(
    const ReticulationConfigSet &left, const ReticulationConfigSet &right) {
  ReticulationConfigSet res(left.max_reticulations);
  for (size_t i = 0; i < left.configs.size(); ++i) {
    for (size_t j = 0; j < right.configs.size(); ++j) {
      std::vector<ReticulationState> combined =
          combineReticulationChoices(left.configs[i], right.configs[j]);
      if (validReticulationChoices(combined)) {
        res.configs.emplace_back(combined);
      }
    }
  }
  simplifyReticulationChoices(res);
  return res;
}

}  // namespace netrax