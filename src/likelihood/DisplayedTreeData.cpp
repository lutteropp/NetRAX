#include "DisplayedTreeData.hpp"

#include <stdexcept>
#include <iostream>
#include "mpreal.h"

namespace netrax
{
    bool single_clv_is_all_zeros(ClvRangeInfo rangeInfo, double* clv) {
        for (size_t i = 0; i < rangeInfo.inner_clv_num_entries; ++i) {
            if (clv[i] != 0.0) {
                return false;
            }
        }
        return true;
    }

    void print_node_scaler(ScaleBufferRangeInfo rangeInfo, unsigned int * scale_buffer) {
        for (size_t j = 0; j < rangeInfo.scaler_size; ++j) {
            std::cout << scale_buffer[j] << ", ";
        }
        std::cout << "\n";
    }

    void print_node_clv(ClvRangeInfo rangeInfo, double * clv) {
        for (size_t j = 0; j < rangeInfo.inner_clv_num_entries; ++j) {
            std::cout << clv[j] << ", ";
        }
        std::cout << "\n";
    }

    void print_clv(ClvRangeInfo rangeInfo, double ** clv) {
        for (size_t i = rangeInfo.start; i < rangeInfo.end; ++i) {
            std::cout << "clv[" << i << ":\n";
            for (size_t j = 0; j < rangeInfo.inner_clv_num_entries; ++j) {
                std::cout << clv[i][j] << ", ";
            }
            std::cout << "\n";
        }
    }

    ClvRangeInfo get_clv_range(pll_partition_t* partition) {
        assert(partition);
        assert(!pll_repeats_enabled(partition));

        unsigned int alignment = partition->alignment;
        unsigned int total_num_clvs = partition->nodes;
         /* if tip pattern precomputation is enabled, then do not allocate CLV space
    for the tip nodes */
        unsigned int start = (partition->attributes & PLL_ATTRIB_PATTERN_TIP) ? partition->tips : 0;

        unsigned int end = partition->clv_buffers + partition->tips;
        
        unsigned int sites_alloc = (unsigned int)partition->asc_additional_sites + partition->sites;
        size_t inner_clv_num_entries = (size_t)sites_alloc * partition->states_padded * partition->rate_cats;
        
        return ClvRangeInfo{alignment, total_num_clvs, start, end, inner_clv_num_entries};
    }

    double* create_single_empty_clv(ClvRangeInfo rangeInfo) {
        double* single_clv = (double *)pll_aligned_alloc(rangeInfo.inner_clv_num_entries * sizeof(double), rangeInfo.alignment);
        if (!single_clv)
        {
            throw std::runtime_error("Unable to allocate enough memory for CLVs.");
        }
        /* zero-out CLV vectors to avoid valgrind warnings when using odd number of
        states with vectorized code */
        memset(single_clv,
                0,
                rangeInfo.inner_clv_num_entries * sizeof(double));
        return single_clv;
    }

    double** create_empty_clv_vector(ClvRangeInfo rangeInfo) {
        double **clv = (double **)calloc(rangeInfo.total_num_clvs, sizeof(double *));
        if (!clv)
        {
            throw std::runtime_error("Unable to allocate enough memory for CLVs.");
        }

        for (unsigned int i = rangeInfo.start; i < rangeInfo.end; ++i)
        {
            clv[i] = create_single_empty_clv(rangeInfo);
        }
        return clv;
    }

    bool clv_single_entries_equal(ClvRangeInfo rangeInfo, double* clv1, double* clv2) {
        if (clv1 == clv2) {
            return true;
        }
        for (size_t j = 0; j < rangeInfo.inner_clv_num_entries; ++j) {
            if (clv1[j] != clv2[j]) {
                return false;
            }
        }
        return true;
    }

    bool clv_entries_equal(ClvRangeInfo rangeInfo, double** clv1, double** clv2) {
        for (unsigned int i = rangeInfo.start; i < rangeInfo.end; ++i)
        {
            for (size_t j = 0; j < rangeInfo.inner_clv_num_entries; ++j) {
                if (clv1[i][j] != clv2[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    void assign_clv_entries(pll_partition_t* partition, double** from_clv, double** to_clv) {
        assert(from_clv);
        assert(to_clv);
        assert(partition);

        ClvRangeInfo rangeInfo = get_clv_range(partition);

        for (unsigned int i = rangeInfo.start; i < rangeInfo.end; ++i)
        {
            memcpy(to_clv[i],
                   from_clv[i],
                   rangeInfo.inner_clv_num_entries * sizeof(double));
        }
    }

    double* clone_single_clv_vector(ClvRangeInfo clvInfo, double* clv) {
        if (!clv) {
            return nullptr;
        }
        double *cloned_clv = create_single_empty_clv(clvInfo);
        memcpy(cloned_clv, clv, clvInfo.inner_clv_num_entries * sizeof(double));
        return cloned_clv;
    }

    double** clone_clv_vector(pll_partition_t *partition, double** clv) {
        assert(partition);
        double **cloned_clv = create_empty_clv_vector(get_clv_range(partition));
        assign_clv_entries(partition, clv, cloned_clv);
        return cloned_clv;
    }

    void delete_cloned_clv_vector(ClvRangeInfo rangeInfo, double **clv) {
        if (clv)
        {
            for (unsigned int i = rangeInfo.start; i < rangeInfo.end; ++i)
                pll_aligned_free(clv[i]);
        }
        free(clv);
    }

    void delete_cloned_clv_vector(pll_partition_t *partition, double **clv) {
        assert(partition);
        delete_cloned_clv_vector(get_clv_range(partition), clv);
    }

    ScaleBufferRangeInfo get_scale_buffer_range(pll_partition_t* partition) {
        assert(partition);
        assert(!pll_repeats_enabled(partition));

        unsigned int sites_alloc = (unsigned int) partition->asc_additional_sites + partition->sites;
        unsigned int scaler_size = (partition->attributes & PLL_ATTRIB_RATE_SCALERS) ? sites_alloc * partition->rate_cats : sites_alloc;
        unsigned int num_scale_buffers = partition->scale_buffers;
        return ScaleBufferRangeInfo{scaler_size, num_scale_buffers};
    }

    unsigned int* create_single_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo) {
        unsigned int* single_scale_buffer = (unsigned int *) calloc(rangeInfo.scaler_size, sizeof(unsigned int));
        if (!single_scale_buffer) {
            throw std::runtime_error("Unable to allocate enough memory for scale buffer.");
        }
        return single_scale_buffer;
    }

    unsigned int** create_empty_scale_buffer(ScaleBufferRangeInfo rangeInfo) {
        unsigned int ** scale_buffer = (unsigned int **) calloc(rangeInfo.num_scale_buffers, sizeof(unsigned int *));

        if (!scale_buffer) {
             throw std::runtime_error("Unable to allocate enough memory for scale buffer.");
        }

        for (unsigned int i = 0; i < rangeInfo.num_scale_buffers; ++i) {
            scale_buffer[i] = create_single_empty_scale_buffer(rangeInfo);
        }
        return scale_buffer;
    }

    bool scale_buffer_single_entries_equal(ScaleBufferRangeInfo rangeInfo, unsigned int* scale_buffer_1, unsigned int* scale_buffer_2) {
        if (scale_buffer_1 == scale_buffer_2) {
            return true;
        }
        for (size_t j = 0; j < rangeInfo.scaler_size; ++j) {
            if (scale_buffer_1[j] != scale_buffer_2[j]) {
                return false;
            }
        }
        return true;
    }

    bool scale_buffer_entries_equal(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer_1, unsigned int** scale_buffer_2) {
        for (unsigned int i = 0; i < rangeInfo.num_scale_buffers; ++i)
        {
            for (size_t j = 0; j < rangeInfo.scaler_size; ++j) {
                if (scale_buffer_1[i][j] != scale_buffer_2[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    unsigned int* clone_single_scale_buffer(ScaleBufferRangeInfo scaleBufferInfo, unsigned int* scale_buffer) {
        if (!scale_buffer) {
            return nullptr;
        }
        unsigned int *cloned_scale_buffer = create_single_empty_scale_buffer(scaleBufferInfo);
        memcpy(cloned_scale_buffer, scale_buffer, scaleBufferInfo.scaler_size * sizeof(unsigned int));
        return cloned_scale_buffer;
    }

    unsigned int** clone_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer) {
        unsigned int **cloned_scale_buffer = create_empty_scale_buffer(get_scale_buffer_range(partition));
        assign_scale_buffer_entries(partition, scale_buffer, cloned_scale_buffer);
        return cloned_scale_buffer;
    }

    void delete_cloned_scale_buffer(ScaleBufferRangeInfo rangeInfo, unsigned int** scale_buffer) {
        if (scale_buffer) {
            for (unsigned int i = 0; i < rangeInfo.num_scale_buffers; ++i) {
                free(scale_buffer[i]);
            }
        }
    }

    void delete_cloned_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer) {
        delete_cloned_scale_buffer(get_scale_buffer_range(partition), scale_buffer);
    }

    void assign_scale_buffer_entries(pll_partition_t* partition, unsigned int** from_scale_buffer, unsigned int** to_scale_buffer) {
        assert(from_scale_buffer);
        assert(to_scale_buffer);
        ScaleBufferRangeInfo rangeInfo = get_scale_buffer_range(partition);
        for (unsigned int i = 0; i < rangeInfo.num_scale_buffers; ++i)
        {
            memcpy(to_scale_buffer[i],
                   from_scale_buffer[i],
                   rangeInfo.scaler_size * sizeof(unsigned int));
        }
    }

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

    void printReticulationChoices(const std::vector<ReticulationState>& reticulationChoices) {
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

    mpfr::mpreal computeReticulationChoicesLogProb_internal(const std::vector<ReticulationState>& choices, const std::vector<double>& reticulationProbs) {
        mpfr::mpreal logProb = 0;
        for (size_t i = 0; i < reticulationProbs.size(); ++i) {
            if (choices[i] != ReticulationState::DONT_CARE) {
                mpfr::mpreal prob;
                if (choices[i] == ReticulationState::TAKE_FIRST_PARENT) {
                    prob = reticulationProbs[i];
                } else {
                    prob = 1.0 - reticulationProbs[i];
                }
                logProb += mpfr::log(prob);
            }
        }
        return logProb;
    }

    double computeReticulationChoicesLogProb(const std::vector<ReticulationState>& choices, const std::vector<double>& reticulationProbs) {
        return computeReticulationChoicesLogProb_internal(choices, reticulationProbs).toDouble();
    }

    double computeReticulationConfigLogProb(const ReticulationConfigSet& choices, const std::vector<double>& reticulationProbs) {
        mpfr::mpreal prob = 0.0;
        for (size_t i = 0; i < choices.configs.size(); ++i) {
            prob += mpfr::exp(computeReticulationChoicesLogProb_internal(choices.configs[i], reticulationProbs));
        }
        return mpfr::log(prob).toDouble();
    }

    double computeReticulationConfigProb(const ReticulationConfigSet& choices, const std::vector<double>& reticulationProbs) {
        mpfr::mpreal prob = 0.0;
        for (size_t i = 0; i < choices.configs.size(); ++i) {
            prob += mpfr::exp(computeReticulationChoicesLogProb_internal(choices.configs[i], reticulationProbs));
        }
        return prob.toDouble();
    }

    //bool reticulationChoicesCompatible(const std::vector<ReticulationState>& left, const std::vector<ReticulationState>& right);
    bool reticulationConfigsCompatible(const ReticulationConfigSet& left, const ReticulationConfigSet& right) {
        for (size_t i = 0; i < left.configs.size(); ++i) {
            for (size_t j = 0; j < right.configs.size(); ++j) {
                if (reticulationChoicesCompatible(left.configs[i], right.configs[j])) {
                    return true;
                }
            }
        }
        return false;
    }

    void printReticulationChoices(const ReticulationConfigSet& reticulationChoices) {
        for (size_t i = 0; i < reticulationChoices.configs.size(); ++i) {
            printReticulationChoices(reticulationChoices.configs[i]);
        }
    }

    void simplifyReticulationChoices(ReticulationConfigSet& res) {
        // simplify the reticulation choices. E.g., if we have both 00 and 01, summarize them to 0-
        bool shrinked = true;
        while (shrinked) {
            shrinked = false;

            // fist: search for duplicates
            for (size_t i = 0; i < res.configs.size(); ++i) {
                for (size_t j = i+1; j < res.configs.size(); ++j) {
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

    ReticulationConfigSet combineReticulationChoices(const ReticulationConfigSet& left, const ReticulationConfigSet& right) {
        ReticulationConfigSet res(left.max_reticulations);
        for (size_t i = 0; i < left.configs.size(); ++i) {
            for (size_t j = 0; j < right.configs.size(); ++j) {
                if (reticulationChoicesCompatible(left.configs[i], right.configs[j])) {
                    res.configs.emplace_back(combineReticulationChoices(left.configs[i], right.configs[j]));
                }
            }
        }
        simplifyReticulationChoices(res);
        return res;
    }
}