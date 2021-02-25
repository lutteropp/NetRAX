#include "DisplayedTreeData.hpp"

#include <stdexcept>
#include <iostream>

namespace netrax
{
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
         /* if tip pattern precomputation is enabled, then do not allocate CLV space
    for the tip nodes */
        unsigned int start = (partition->attributes & PLL_ATTRIB_PATTERN_TIP) ? partition->tips : 0;

        unsigned int end = partition->clv_buffers + partition->tips;
        
        unsigned int sites_alloc = (unsigned int)partition->asc_additional_sites + partition->sites;
        size_t inner_clv_num_entries = (size_t)sites_alloc * partition->states_padded * partition->rate_cats;
        
        return ClvRangeInfo{start, end, inner_clv_num_entries};
    }

    double **create_empty_clv_vector(pll_partition_t *partition)
    {
        assert(!pll_repeats_enabled(partition));
        double **clv = (double **)calloc(partition->nodes, sizeof(double *));
        if (!clv)
        {
            throw std::runtime_error("Unable to allocate enough memory for CLVs.");
        }
       
        ClvRangeInfo rangeInfo = get_clv_range(partition);

        for (unsigned int i = rangeInfo.start; i < rangeInfo.end; ++i)
        {
            clv[i] = (double *)pll_aligned_alloc(rangeInfo.inner_clv_num_entries * sizeof(double),
                                                 partition->alignment);
            if (!clv[i])
            {
                throw std::runtime_error("Unable to allocate enough memory for CLVs.");
            }
            /* zero-out CLV vectors to avoid valgrind warnings when using odd number of
        states with vectorized code */
            memset(clv[i],
                   0,
                   rangeInfo.inner_clv_num_entries * sizeof(double));
        }
        return clv;
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

        ClvRangeInfo rangeInfo = get_clv_range(partition);

        for (unsigned int i = rangeInfo.start; i < rangeInfo.end; ++i)
        {
            memcpy(to_clv[i],
                   from_clv[i],
                   rangeInfo.inner_clv_num_entries * sizeof(double));
        }
    }

    double **clone_clv_vector(pll_partition_t *partition, double** clv)
    {
        double **cloned_clv = create_empty_clv_vector(partition);
        assign_clv_entries(partition, clv, cloned_clv);
        return cloned_clv;
    }

    void delete_cloned_clv_vector(ClvRangeInfo rangeInfo, double **clv)
    {
        if (clv)
        {
            for (unsigned int i = rangeInfo.start; i < rangeInfo.end; ++i)
                pll_aligned_free(clv[i]);
        }
        free(clv);
    }

    void delete_cloned_clv_vector(pll_partition_t *partition, double **clv)
    {
        delete_cloned_clv_vector(get_clv_range(partition), clv);
    }

    ScaleBufferRangeInfo get_scale_buffer_range(pll_partition_t* partition) {
        unsigned int sites_alloc = (unsigned int) partition->asc_additional_sites + partition->sites;
        unsigned int scaler_size = (partition->attributes & PLL_ATTRIB_RATE_SCALERS) ? sites_alloc * partition->rate_cats : sites_alloc;
        unsigned int num_scale_buffers = partition->scale_buffers;
        return ScaleBufferRangeInfo{scaler_size, num_scale_buffers};
    }

    unsigned int ** create_empty_scale_buffer(pll_partition_t *partition)
    {
        assert(!pll_repeats_enabled(partition));
        unsigned int ** scale_buffer = (unsigned int **) calloc(partition->scale_buffers, sizeof(unsigned int *));

        if (!scale_buffer) {
             throw std::runtime_error("Unable to allocate enough memory for scale buffer.");
        }

        ScaleBufferRangeInfo rangeInfo = get_scale_buffer_range(partition);

        for (unsigned int i = 0; i < partition->scale_buffers; ++i) {
            scale_buffer[i] = (unsigned int *) calloc(rangeInfo.scaler_size, sizeof(unsigned int));
            if (!scale_buffer[i]) {
                throw std::runtime_error("Unable to allocate enough memory for scale buffer.");
            }
        }
        return scale_buffer;
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

    unsigned int** clone_scale_buffer(pll_partition_t* partition, unsigned int** scale_buffer) {
        unsigned int **cloned_scale_buffer = create_empty_scale_buffer(partition);
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
            memcpy(from_scale_buffer[i],
                   to_scale_buffer[i],
                   rangeInfo.scaler_size * sizeof(unsigned int));
        }
    }
}