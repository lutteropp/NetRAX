#include "NetworkDistances.hpp"

#include "graph/NetworkFunctions.hpp"
#include "graph/NetworkTopology.hpp"

#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <unordered_map>

namespace netrax
{

    std::vector<bool> edge_split(AnnotatedNetwork &ann_network, Edge *edge, std::unordered_map<std::string, unsigned int> &label_to_int, bool unrooted, bool softwired)
    {
        Network &network = ann_network.network;
        std::vector<bool> split(network.num_tips(), false);
        // activate everything below the edge
        std::queue<Node *> q;
        q.emplace(getTarget(network, edge));
        while (!q.empty())
        {
            Node *act_node = q.front();
            q.pop();
            if (act_node->isTip())
            {
                if (label_to_int.find(act_node->label) == label_to_int.end())
                {
                    throw std::runtime_error("Unknown taxon name: " + act_node->label);
                }
                split[label_to_int[act_node->label]] = true;
            }
            else
            {
                std::vector<netrax::Node *> children;
                if (softwired)
                {
                    children = getActiveChildren(network, act_node);
                }
                else
                {
                    children = getChildren(network, act_node);
                }

                for (Node *child : children)
                {
                    q.emplace(child);
                }
            }
        }

        if (unrooted)
        {
            // normalization: ensure that we have zero at the first position
            if (split[0] == true)
            {
                for (size_t i = 0; i < split.size(); ++i)
                {
                    split[i] = !split[i];
                }
            }
        }

        return split;
    }

    bool is_trivial_split(const std::vector<bool> &split)
    {
        unsigned int cnt_ones = 0;
        for (size_t i = 0; i < split.size(); ++i)
        {
            if (split[i])
            {
                cnt_ones++;
            }
        }
        return ((cnt_ones == 1) || (cnt_ones == split.size() - 1));
    }

    void add_splits(std::unordered_set<std::vector<bool>> &splits_hash, AnnotatedNetwork &ann_network, std::unordered_map<std::string, unsigned int> &label_to_int, bool unrooted, bool softwired)
    {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i)
        {
            std::vector<bool> act_split = edge_split(ann_network, ann_network.network.edges_by_index[i], label_to_int, unrooted, softwired);
            if (!is_trivial_split(act_split))
            {
                splits_hash.emplace(act_split);
            }
        }
    }

    std::unordered_set<std::vector<bool>> extract_network_splits(AnnotatedNetwork &ann_network, std::unordered_map<std::string, unsigned int> &label_to_int, bool unrooted, bool softwired)
    {
        std::unordered_set<std::vector<bool>> splits_hash;
        if (softwired)
        {
            for (int tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index)
            {
                netrax::setReticulationParents(ann_network.network, tree_index);
                add_splits(splits_hash, ann_network, label_to_int, unrooted, softwired);
            }
        }
        else
        {
            add_splits(splits_hash, ann_network, label_to_int, unrooted, softwired);
        }
        return splits_hash;
    }

    unsigned int count_not_in_other(const std::unordered_set<std::vector<bool>> &splits_hash, const std::unordered_set<std::vector<bool>> &other_splits_hash)
    {
        unsigned int cnt = 0;
        for (const std::vector<bool> &split : splits_hash)
        {
            if (other_splits_hash.count(split) == 0)
            {
                cnt++;
            }
        }
        return cnt;
    }

    unsigned int count_in_both(const std::unordered_set<std::vector<bool>> &splits_hash, const std::unordered_set<std::vector<bool>> &other_splits_hash)
    {
        unsigned int cnt = 0;
        for (const std::vector<bool> &split : splits_hash)
        {
            if (other_splits_hash.count(split) > 0)
            {
                cnt++;
            }
        }
        return cnt;
    }

    double cluster_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork &ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int, bool unrooted, bool softwired)
    {
        std::unordered_set<std::vector<bool>> splits_hash_1 = extract_network_splits(ann_network_1, label_to_int, unrooted, softwired);
        std::unordered_set<std::vector<bool>> splits_hash_2 = extract_network_splits(ann_network_2, label_to_int, unrooted, softwired);
        unsigned int n_1 = splits_hash_1.size();
        unsigned int n_2 = splits_hash_2.size();
        unsigned int n_both = count_in_both(splits_hash_1, splits_hash_2);

        double dist = (double) (n_1 + n_2 - 2 * n_both) / (n_1 + n_2 - n_both);
        return dist;
    }

    double displayed_trees_distance(AnnotatedNetwork& ann_network_1, AnnotatedNetwork& ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int, bool unrooted) {
        unsigned int n_trees_1 = 1 << ann_network_1.network.num_reticulations();
        unsigned int n_trees_2 = 1 << ann_network_2.network.num_reticulations();

        std::vector<std::unordered_set<std::vector<bool>> > tree_splits_1(n_trees_1);
        std::vector<std::unordered_set<std::vector<bool>> > tree_splits_2(n_trees_2);
        
        for (int tree_index_1 = 0; tree_index_1 < 1 << ann_network_1.network.num_reticulations(); ++tree_index_1) {
            netrax::setReticulationParents(ann_network_1.network, tree_index_1);
            add_splits(tree_splits_1[tree_index_1], ann_network_1, label_to_int, unrooted, true);
        }

        for (int tree_index_2 = 0; tree_index_2 < 1 << ann_network_2.network.num_reticulations(); ++tree_index_2) {
            netrax::setReticulationParents(ann_network_2.network, tree_index_2);
            add_splits(tree_splits_2[tree_index_2], ann_network_2, label_to_int, unrooted, true);
        }

        unsigned int n_trees_both = 0;
        for (unsigned int tree_index_1 = 0; tree_index_1 < n_trees_1; ++tree_index_1) {
            bool found_equal_tree = false;
            for (unsigned int tree_index_2 = 0; tree_index_2 < n_trees_2; ++tree_index_2) {
                bool trees_equal = (count_in_both(tree_splits_1[tree_index_1], tree_splits_2[tree_index_2]) == tree_splits_1.size());
                if (trees_equal) {
                    found_equal_tree = true;
                    break;
                }
            }
            if (found_equal_tree) {
                n_trees_both++;
            }
        }

        double dist = (double) (n_trees_1 + n_trees_2 - 2 * n_trees_both) / (n_trees_1 + n_trees_2 - n_trees_both);
        return dist;
    }

    double get_network_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork &ann_network_2, NetworkDistanceType type)
    {
        if (ann_network_1.network.num_tips() != ann_network_2.network.num_tips())
        {
            throw std::runtime_error("Unequal number of taxa");
        }
        std::unordered_map<std::string, unsigned int> label_to_int;
        for (size_t i = 0; i < ann_network_1.network.num_tips(); ++i)
        {
            label_to_int[ann_network_1.network.nodes_by_index[i]->label] = i;
        }

        switch (type)
        {
        case NetworkDistanceType::UNROOTED_SOFTWIRED_DISTANCE:
            return cluster_distance(ann_network_1, ann_network_2, label_to_int, true, true);
        case NetworkDistanceType::ROOTED_SOFTWIRED_DISTANCE:
            return cluster_distance(ann_network_1, ann_network_2, label_to_int, false, true);
        case NetworkDistanceType::UNROOTED_HARDWIRED_DISTANCE:
            return cluster_distance(ann_network_1, ann_network_2, label_to_int, true, false);
        case NetworkDistanceType::ROOTED_HARDWIRED_DISTANCE:
            return cluster_distance(ann_network_1, ann_network_2, label_to_int, false, false);
        case NetworkDistanceType::UNROOTED_DISPLAYED_TREES_DISTANCE:
            return displayed_trees_distance(ann_network_1, ann_network_2, label_to_int, true);
        case NetworkDistanceType::ROOTED_DISPLAYED_TREES_DISTANCE:
            return displayed_trees_distance(ann_network_1, ann_network_2, label_to_int, false);
        default:
            throw std::runtime_error("Required network distance type not implemented yet!");
        }
    }
} // namespace netrax