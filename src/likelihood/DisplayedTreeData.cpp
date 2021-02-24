#include "DisplayedTreeData.hpp"

#include <stdexcept>

namespace netrax
{
    double **create_empty_clv_vector(pll_partition_t *partition)
    {
        double **clv = (double **)calloc(partition->nodes, sizeof(double *));
        if (!clv)
        {
            throw std::runtime_error("Unable to allocate enough memory for CLVs.");
        }
        /* if tip pattern precomputation is enabled, then do not allocate CLV space
    for the tip nodes */
        unsigned int start = (partition->attributes & PLL_ATTRIB_PATTERN_TIP) ? partition->tips : 0;
        unsigned int sites_alloc = (unsigned int)partition->asc_additional_sites + partition->sites;

        for (unsigned int i = start; i < partition->tips + partition->clv_buffers; ++i)
        {
            clv[i] = (double *)pll_aligned_alloc(sites_alloc * partition->states_padded *
                                                     partition->rate_cats * sizeof(double),
                                                 partition->alignment);
            if (!clv[i])
            {
                throw std::runtime_error("Unable to allocate enough memory for CLVs.");
            }
            /* zero-out CLV vectors to avoid valgrind warnings when using odd number of
        states with vectorized code */
            memset(clv[i],
                   0,
                   (size_t)sites_alloc * partition->states_padded * partition->rate_cats * sizeof(double));
        }
        return clv;
    }

    double **clone_clv_vector(pll_partition_t *partition)
    {
        double **cloned_clv = create_empty_clv_vector(partition);

        unsigned int start = (partition->attributes & PLL_ATTRIB_PATTERN_TIP) ? partition->tips : 0;
        unsigned int sites_alloc = (unsigned int)partition->asc_additional_sites + partition->sites;

        for (unsigned int i = start; i < partition->tips + partition->clv_buffers; ++i)
        {
            memcpy(cloned_clv[i],
                   partition->clv[i],
                   (size_t)sites_alloc * partition->states_padded * partition->rate_cats * sizeof(double));
        }

        return cloned_clv;
    }

    void delete_cloned_clv_vector(pll_partition_t *partition, double **clv)
    {
        if (clv)
        {
            unsigned int start = (partition->attributes & PLL_ATTRIB_PATTERN_TIP) ? partition->tips : 0;
            for (unsigned int i = start; i < partition->clv_buffers + partition->tips; ++i)
                pll_aligned_free(clv[i]);
        }
        free(clv);
    }
}