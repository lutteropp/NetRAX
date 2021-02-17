#include "NetworkDistances.hpp"

#include "graph/NetworkFunctions.hpp"
#include "graph/NetworkTopology.hpp"

#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <unordered_map>

namespace netrax {

    std::vector<bool> edge_split(AnnotatedNetwork& ann_network, Edge* edge, std::unordered_map<std::string, unsigned int>& label_to_int, bool unrooted, bool softwired) {
        Network& network = ann_network.network;
        std::vector<bool> split(network.num_tips(), false);
        // activate everything below the edge
        std::queue<Node*> q;
        q.emplace(getTarget(network, edge));
        while (!q.empty()) {
            Node* act_node = q.front();
            q.pop();
            if (act_node->isTip()) {
                if (label_to_int.find(act_node->label) == label_to_int.end()) {
                    throw std::runtime_error("Unknown taxon name: " + act_node->label);
                }
                split[label_to_int[act_node->label]] = true;
            } else {
                std::vector<netrax::Node*> children;
                if (softwired) {
                    children = getActiveChildren(network, act_node);
                } else {
                    children = getChildren(network, act_node);
                }
                
                for (Node* child : children) {
                    q.emplace(child);
                }
            }
        }
        
        if (unrooted) {
            // normalization: ensure that we have zero at the first position
            if (split[0] == true) {
                for (size_t i = 0; i < split.size(); ++i) {
                    split[i] = !split[i];
                }
            }
        }

        return split;
    }

    bool is_trivial_split(const std::vector<bool>& split) {
        unsigned int cnt_ones = 0;
        for (size_t i = 0; i < split.size(); ++i) {
            if (split[i]) {
                cnt_ones++;
            }
        }
        return ((cnt_ones == 1) || (cnt_ones == split.size() - 1));
    }

    void add_splits(std::unordered_set<std::vector<bool> >& splits_hash, AnnotatedNetwork& ann_network, std::unordered_map<std::string, unsigned int>& label_to_int, bool unrooted, bool softwired) {
        for (size_t i = 0; i < ann_network.network.num_branches(); ++i) {
            std::vector<bool> act_split = edge_split(ann_network, ann_network.network.edges_by_index[i], label_to_int, unrooted, softwired);
            if (!is_trivial_split(act_split)) {
                splits_hash.emplace(act_split);
            }
        }
    }

    std::unordered_set<std::vector<bool> > extract_network_splits(AnnotatedNetwork& ann_network, std::unordered_map<std::string, unsigned int>& label_to_int, bool unrooted, bool softwired) {
        std::unordered_set<std::vector<bool> > splits_hash;
        if (softwired) {
            for (int tree_index = 0; tree_index < 1 << ann_network.network.num_reticulations(); ++tree_index) {
                netrax::setReticulationParents(ann_network.network, tree_index);
                add_splits(splits_hash, ann_network, label_to_int, unrooted, softwired);
            }
        } else {
            add_splits(splits_hash, ann_network, label_to_int, unrooted, softwired);
        }
        return splits_hash;
    }

    unsigned int count_not_in_other(const std::unordered_set<std::vector<bool> >& splits_hash, const std::unordered_set<std::vector<bool> >& other_splits_hash) {
        unsigned int cnt = 0;
        for (const std::vector<bool>& split : splits_hash) {
            if (other_splits_hash.count(split) == 0)
            {
                cnt++;
            }
        }
        return cnt;
    }

    unsigned int count_in_both(const std::unordered_set<std::vector<bool> >& splits_hash, const std::unordered_set<std::vector<bool> >& other_splits_hash) {
        unsigned int cnt = 0;
        for (const std::vector<bool>& split : splits_hash) {
            if (other_splits_hash.count(split) > 0)
            {
                cnt++;
            }
        }
        return cnt;
    }

    double get_network_distance(AnnotatedNetwork& ann_network_1, AnnotatedNetwork& ann_network_2, NetworkDistanceType type) {
        if (ann_network_1.network.num_tips() != ann_network_2.network.num_tips()) {
            throw std::runtime_error("Unequal number of taxa");
        }
        std::unordered_map<std::string, unsigned int> label_to_int;
        for (size_t i = 0; i < ann_network_1.network.num_tips(); ++i) {
            label_to_int[ann_network_1.network.nodes_by_index[i]->label] = i;
        }

        bool unrooted = true;
        if ((type == NetworkDistanceType::ROOTED_SOFTWIRED_DISTANCE) || (type == NetworkDistanceType::ROOTED_HARDWIRED_DISTANCE)) {
            unrooted = false;
        }
        bool softwired = true;
        if ((type == NetworkDistanceType::ROOTED_HARDWIRED_DISTANCE) || (type == NetworkDistanceType::UNROOTED_HARDWIRED_DISTANCE)) {
            softwired = false;
        }

        std::unordered_set<std::vector<bool> > splits_hash_1 = extract_network_splits(ann_network_1, label_to_int, unrooted, softwired);
        std::unordered_set<std::vector<bool> > splits_hash_2 = extract_network_splits(ann_network_2, label_to_int, unrooted, softwired);

        unsigned int one_but_not_two = count_not_in_other(splits_hash_1, splits_hash_2);
        unsigned int two_but_not_one = count_not_in_other(splits_hash_2, splits_hash_1);

        double dist = (double) (one_but_not_two + two_but_not_one) / (splits_hash_1.size() + splits_hash_2.size() - count_in_both(splits_hash_1, splits_hash_2));
        return dist;
    }
}