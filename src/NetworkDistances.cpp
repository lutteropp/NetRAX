#include "NetworkDistances.hpp"

#include "helper/Helper.hpp"

#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <functional>

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

    std::vector<int> edge_trip(AnnotatedNetwork &ann_network, Edge *edge, std::unordered_map<std::string, unsigned int> &label_to_int)
    {
        Network &network = ann_network.network;
        std::vector<unsigned int> descendant_in_trees_count(network.num_tips(), 0); // first, we count in how many displayed trees a taxon is a descendant
        unsigned int n_trees = 1 << ann_network.network.num_reticulations();

        for (unsigned int tree_index = 0; tree_index < n_trees; ++tree_index)
        {
            netrax::setReticulationParents(ann_network.network, tree_index);
            // increase everything below the edge
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
                    descendant_in_trees_count[label_to_int[act_node->label]]++;
                }
                else
                {
                    std::vector<netrax::Node *> children = getActiveChildren(network, act_node);
                    for (Node *child : children)
                    {
                        q.emplace(child);
                    }
                }
            }
        }

        std::vector<int> trip(network.num_tips(), 0);
        for (size_t i = 0; i < trip.size(); ++i) {
            if (descendant_in_trees_count[i] > 0) {
                if (descendant_in_trees_count[i] == n_trees) {
                    trip[i] = 1; // stable descendant
                } else {
                    trip[i] = 2; // unstable descendant
                }
            }
        }

        return trip;
    }


    int count_paths(AnnotatedNetwork &ann_network, Node* source, Node* target) {
        if (source == target) {
            return 1;
        }
        if (source->isTip()) {
            return 0;
        }
        int n_paths = 0;
        auto children = getChildren(ann_network.network, source);
        for (auto& child : children) {
            n_paths += count_paths(ann_network, child, target);
        }
        return n_paths;
    }


    std::vector<int> node_path_vector(AnnotatedNetwork &ann_network, Node *node, std::unordered_map<std::string, unsigned int> &label_to_int) {
        std::vector<int> path_vector(ann_network.network.num_tips(), 0);
        for (size_t i = 0; i < ann_network.network.num_tips(); ++i) {
            unsigned int tip_id = label_to_int[ann_network.network.nodes_by_index[i]->label];
            path_vector[tip_id] = count_paths(ann_network, node, ann_network.network.nodes_by_index[i]);
        }
        return path_vector;
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

    bool is_trivial_trip(const std::vector<int> &trip)
    {
        unsigned int cnt_zeros = 0; // these are the non-decendants of the edge associated with the given tripartition
        for (size_t i = 0; i < trip.size(); ++i)
        {
            if (trip[i] == 0)
            {
                cnt_zeros++;
            }
        }
        return ((cnt_zeros == 1) || (cnt_zeros == trip.size() - 1));
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

    struct VectorHash {
        size_t operator()(const std::vector<int>& v) const {
            std::hash<int> hasher;
            size_t seed = 0;
            for (int i : v) {
                seed ^= hasher(i) + 0x9e3779b9 + (seed<<6) + (seed>>2);
            }
            return seed;
        }
    };

    struct NestedLabel {
        std::string newick;

        NestedLabel() {
            newick = "";
        }

        NestedLabel(std::vector<NestedLabel>& childrenLabels) {
            newick = "(";
            std::vector<std::string> childrenStrings;
            for (size_t i = 0; i < childrenLabels.size(); ++i) {
                childrenStrings.emplace_back(childrenLabels[i].newick);
            }
            std::sort(childrenStrings.begin(), childrenStrings.end());

            for (size_t i = 0; i < childrenStrings.size(); ++i) {
                newick += childrenStrings[i];
                if (i + 1 < childrenStrings.size()) {
                    newick += ",";
                }
            }
            newick += ")";
        }

        NestedLabel(int tip_id) {
            newick = std::to_string(tip_id);
        }

        bool empty() const {
            return newick.empty();
        }

        bool operator==(const NestedLabel& other) const {
            return newick == other.newick;
        }
    };

    struct NestedLabelHash {
        size_t operator()(const NestedLabel& v) const {
            return std::hash<std::string>{}(v.newick);
        }
    };

    std::unordered_set<std::vector<int>, VectorHash> extract_network_trips(AnnotatedNetwork &ann_network, std::unordered_map<std::string, unsigned int> &label_to_int)
    {
        std::unordered_set<std::vector<int>, VectorHash> trips_hash;
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i)
        {
            std::vector<int> act_trip = edge_trip(ann_network, ann_network.network.edges_by_index[i], label_to_int);
            if (!is_trivial_trip(act_trip))
            {
                trips_hash.emplace(act_trip);
            }
        }
        return trips_hash;
    }

    std::unordered_set<std::vector<int>, VectorHash> extract_network_path_vectors(AnnotatedNetwork& ann_network, std::unordered_map<std::string, unsigned int> &label_to_int) {
        std::unordered_set<std::vector<int>, VectorHash> path_vector_hash;
        for (size_t i = 0; i < ann_network.network.num_nodes(); ++i)
        {
            std::vector<int> act_path_vector = node_path_vector(ann_network, ann_network.network.nodes_by_index[i], label_to_int);
            path_vector_hash.emplace(act_path_vector);
        }
        return path_vector_hash;
    }

    void update_node_nested_labels(AnnotatedNetwork& ann_network, unsigned int node_idx, std::vector<NestedLabel>& nested_labels_by_node, std::unordered_map<std::string, unsigned int> &label_to_int) {
        if (!nested_labels_by_node[node_idx].empty()) {
            return;
        }
        Node* node = ann_network.network.nodes_by_index[node_idx];
        if (node->isTip()) {
            nested_labels_by_node[node_idx] = NestedLabel(label_to_int[node->label]);
            return;
        } else {
            auto children = getChildren(ann_network.network, node);
            std::vector<NestedLabel> childrenLabels;
            for (auto& child : children) {
                if (nested_labels_by_node[child->clv_index].empty()) {
                    update_node_nested_labels(ann_network, child->clv_index, nested_labels_by_node, label_to_int);
                }
                childrenLabels.emplace_back(nested_labels_by_node[child->clv_index]);
            }
            nested_labels_by_node[node_idx] = NestedLabel(childrenLabels);
        }
    }

    std::unordered_set<NestedLabel, NestedLabelHash> extract_network_nested_labels(AnnotatedNetwork& ann_network, std::unordered_map<std::string, unsigned int> &label_to_int) {
        std::unordered_set<NestedLabel, NestedLabelHash> nested_labels_set;
        std::vector<NestedLabel> nested_labels_by_node(ann_network.network.num_nodes());
        
        for (size_t i = 0; i < ann_network.network.num_nodes(); ++i) {
            update_node_nested_labels(ann_network, i, nested_labels_by_node, label_to_int);
            if (i > ann_network.network.num_tips()) { // exclude the trivial tip labels
                nested_labels_set.emplace(nested_labels_by_node[i]);
            }
        }

        return nested_labels_set;
    }

    template <typename T>
    unsigned int count_in_both(T &set_1, T &set_2)
    {
        unsigned int cnt = 0;
        for (const auto &split : set_1)
        {
            if (set_2.count(split) > 0)
            {
                cnt++;
            }
        }
        return cnt;
    }

    template <typename T>
    double relative_distance_score(T& set_1, T& set_2) {
        unsigned int n_1 = set_1.size();
        unsigned int n_2 = set_2.size();
        unsigned int n_both = count_in_both(set_1, set_2);

        double dist = (double)(n_1 + n_2 - 2 * n_both) / (n_1 + n_2 - n_both);
        return dist;
    }

    double rooted_tripartition_distance(AnnotatedNetwork& ann_network_1, AnnotatedNetwork& ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int) {
        auto trips_hash_1 = extract_network_trips(ann_network_1, label_to_int);
        auto trips_hash_2 = extract_network_trips(ann_network_2, label_to_int);
        return relative_distance_score(trips_hash_1, trips_hash_2);
    }

    double cluster_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork &ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int, bool unrooted, bool softwired)
    {
        auto splits_hash_1 = extract_network_splits(ann_network_1, label_to_int, unrooted, softwired);
        auto splits_hash_2 = extract_network_splits(ann_network_2, label_to_int, unrooted, softwired);
        return relative_distance_score(splits_hash_1, splits_hash_2);
    }

    double displayed_trees_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork &ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int, bool unrooted)
    {
        unsigned int n_trees_1 = 1 << ann_network_1.network.num_reticulations();
        unsigned int n_trees_2 = 1 << ann_network_2.network.num_reticulations();

        std::vector<std::unordered_set<std::vector<bool>>> tree_splits_1(n_trees_1);
        std::vector<std::unordered_set<std::vector<bool>>> tree_splits_2(n_trees_2);

        for (int tree_index_1 = 0; tree_index_1 < 1 << ann_network_1.network.num_reticulations(); ++tree_index_1)
        {
            netrax::setReticulationParents(ann_network_1.network, tree_index_1);
            add_splits(tree_splits_1[tree_index_1], ann_network_1, label_to_int, unrooted, true);
        }

        for (int tree_index_2 = 0; tree_index_2 < 1 << ann_network_2.network.num_reticulations(); ++tree_index_2)
        {
            netrax::setReticulationParents(ann_network_2.network, tree_index_2);
            add_splits(tree_splits_2[tree_index_2], ann_network_2, label_to_int, unrooted, true);
        }

        unsigned int n_trees_both = 0;
        for (unsigned int tree_index_1 = 0; tree_index_1 < n_trees_1; ++tree_index_1)
        {
            bool found_equal_tree = false;
            for (unsigned int tree_index_2 = 0; tree_index_2 < n_trees_2; ++tree_index_2)
            {
                bool trees_equal = (count_in_both(tree_splits_1[tree_index_1], tree_splits_2[tree_index_2]) == tree_splits_1.size());
                if (trees_equal)
                {
                    found_equal_tree = true;
                    break;
                }
            }
            if (found_equal_tree)
            {
                n_trees_both++;
            }
        }

        double dist = (double)(n_trees_1 + n_trees_2 - 2 * n_trees_both) / (n_trees_1 + n_trees_2 - n_trees_both);
        return dist;
    }

    double rooted_path_multiplicity_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork& ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int) {
        auto paths_hash_1 = extract_network_path_vectors(ann_network_1, label_to_int);
        auto paths_hash_2 = extract_network_path_vectors(ann_network_2, label_to_int);
        return relative_distance_score(paths_hash_1, paths_hash_2);
    }

    double rooted_nested_labels_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork& ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int) {
        auto labels_hash_1 = extract_network_nested_labels(ann_network_1, label_to_int);
        auto labels_hash_2 = extract_network_nested_labels(ann_network_2, label_to_int);
        return relative_distance_score(labels_hash_1, labels_hash_2);
    }

    double get_network_distance(AnnotatedNetwork &ann_network_1, AnnotatedNetwork &ann_network_2, std::unordered_map<std::string, unsigned int> &label_to_int, NetworkDistanceType type)
    {
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
        case NetworkDistanceType::ROOTED_TRIPARTITION_DISTANCE:
            return rooted_tripartition_distance(ann_network_1, ann_network_2, label_to_int);
        case NetworkDistanceType::ROOTED_PATH_MULTIPLICITY_DISTANCE:
            return rooted_path_multiplicity_distance(ann_network_1, ann_network_2, label_to_int);
        case NetworkDistanceType::ROOTED_NESTED_LABELS_DISTANCE:
            return rooted_nested_labels_distance(ann_network_1, ann_network_2, label_to_int);
        default:
            throw std::runtime_error("Required network distance type not implemented yet!");
        }
    }
} // namespace netrax