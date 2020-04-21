/*
 * ReticulationData.hpp
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include <string>
#include <vector>
#include "Link.hpp"
#include "NodeType.hpp"
#include "NetworkFunctions.hpp"

namespace netrax {

class ReticulationData {
public:
    void init(size_t index, const std::string &label, bool activeParent, Link *linkToFirstParent,
            Link *linkToSecondParent, Link *linkToChild, double prob, size_t num_partitions = 1) {
        this->reticulation_index = index;
        this->label = label;
        this->active_parent_toggle = activeParent;
        this->link_to_first_parent = linkToFirstParent;
        this->link_to_second_parent = linkToSecondParent;
        this->link_to_child = linkToChild;
        this->prob = std::vector<double>(num_partitions, prob);
    }

    size_t getReticulationIndex() const {
        return reticulation_index;
    }
    const std::string& getLabel() const {
        return label;
    }
    Link* getLinkToActiveParent() const {
        if (active_parent_toggle == 0) {
            return link_to_first_parent;
        } else {
            return link_to_second_parent;
        }
    }

    void setActiveParentToggle(bool val) {
        active_parent_toggle = val;
    }
    double getProb(size_t partition = 0) const {
        return prob[partition];
    }
    double getActiveProb(size_t partition = 0) const {
        if (active_parent_toggle == 0) {
            return prob[partition];
        } else {
            return 1.0 - prob[partition];
        }
    }
    void setProb(double val, size_t partition = 0) {
        prob[partition] = val;
    }
    Link* getLinkToFirstParent() const {
        return link_to_first_parent;
    }
    Link* getLinkToSecondParent() const {
        return link_to_second_parent;
    }
    Link* getLinkToChild() const {
        return link_to_child;
    }
    void setLinkToFirstParent(Link *link) {
        link_to_first_parent = link;
    }
    void setLinkToSecondParent(Link *link) {
        link_to_second_parent = link;
    }
    void setLinkToChild(Link *link) {
        link_to_child = link;
    }
    void setNumPartitions(size_t partition_count) {
        size_t old_num_partitions = prob.size();
        prob.resize(partition_count);
        for (size_t i = old_num_partitions; i < partition_count; ++i) {
            prob[i] = prob[0];
        }
    }

    void reset() {
        active_parent_toggle = 0;
        reticulation_index = 0;
        link_to_first_parent = nullptr;
        link_to_second_parent = nullptr;
        link_to_child = nullptr;
        label = "";
        prob = { 0.5 };
    }

    bool active_parent_toggle = 0; // 0: first_parent, 1: second_parent
    size_t reticulation_index = 0;
    Link *link_to_first_parent = nullptr; // The link that has link->outer->node as the first parent
    Link *link_to_second_parent = nullptr; // The link that has link->outer->node as the second parent
    Link *link_to_child = nullptr; // The link that has link->outer->node as the child
    std::string label = "";
    std::vector<double> prob = { 0.5 }; // probability of taking the first parent, for each partition
};

}
