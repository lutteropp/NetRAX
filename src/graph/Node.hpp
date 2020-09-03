/*
 * A node in a phylogenetic network.
 *
 *  Created on: Oct 14, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

#include "ReticulationData.hpp"
#include "Link.hpp"
#include <memory>
#include <cassert>
#include <vector>
#include <algorithm>

namespace netrax {

struct Node {
public:
    void initBasic(size_t index, int scaler_index, const std::string &label) {
        this->clv_index = index;
        this->scaler_index = scaler_index;
        this->label = label;
        this->type = NodeType::BASIC_NODE;
        this->reticulationData = nullptr;
        if (scaler_index >= 0) {
            links.reserve(3);
        } else {
            links.reserve(1);
        }
    }
    void initReticulation(size_t index, int scaler_index, const std::string &label,
            const ReticulationData &retData) {
        this->clv_index = index;
        this->scaler_index = scaler_index;
        this->label = label;
        this->type = NodeType::RETICULATION_NODE;
        reticulationData = std::make_unique<ReticulationData>(retData);
        links.reserve(3);
    }

    bool isTip() const {
        assert(!links.empty());
        return (links.size() == 1);
    }

    void setClvIndex(size_t index) {
        this->clv_index = index;
    }

    void setScalerIndex(int idx) {
        scaler_index = idx;
    }

    Link* getLink() {
        return &links[0];
    }

    Link* addLink(Link &link) {
        links.emplace_back(link);
        assert(links.size() <= 3);

        // update the next pointers
        for (size_t i = 0; i < links.size(); ++i) {
            links[i].next = &links[i + 1 % links.size()];
        }

        return &(links[links.size() - 1]);
    }

    const std::string& getLabel() const {
        return label;
    }
    void setLabel(const std::string &label) {
        this->label = label;
    }
    NodeType getType() const {
        return type;
    }

    const std::unique_ptr<ReticulationData>& getReticulationData() const {
        assert(type == NodeType::RETICULATION_NODE);
        return reticulationData;
    }

    void clear() {
        type = NodeType::BASIC_NODE;
        scaler_index = -1;
        clv_index = 0;
        links.clear();
        if (reticulationData) {
            reticulationData.release();
        }
        reticulationData = nullptr;
        label = "";
    }

    NodeType type = NodeType::BASIC_NODE;
    int scaler_index = -1;
    size_t clv_index = 0;
    std::vector<Link> links;
    std::unique_ptr<ReticulationData> reticulationData = nullptr;
    std::string label = "";
};

}