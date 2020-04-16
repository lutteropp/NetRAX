/*
 * Node.hpp
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
	void initBasic(size_t index, int scaler_index, const std::string& label) {
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
	void initReticulation(size_t index, int scaler_index, const std::string& label, const ReticulationData& retData) {
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

	Link* getLinkToClvIndex(size_t target_index) {
		for (size_t i = 0; i < links.size(); ++i) {
			if (links[i].getTargetNode()->clv_index == target_index) {
				return &links[i];
			}
		}
		return nullptr;
	}

	std::vector<Node*> getChildren(const Node* myParent = nullptr) const {
		if (myParent == nullptr) {
			myParent = getActiveParent();
		}
		std::vector<Node*> children;
		if (type == NodeType::RETICULATION_NODE) {
			children.push_back(reticulationData->getLinkToChild()->getTargetNode());
		} else { // normal node
			std::vector<Node*> neighbors = getNeighbors();
			for (size_t i = 0; i < neighbors.size(); ++i) {
				if (neighbors[i] != myParent) {
					children.push_back(neighbors[i]);
				}
			}
		}
		return children;
	}

	std::vector<Node*> getActiveChildren(const Node* myParent = nullptr) const {
		if (myParent == nullptr) {
			myParent = getActiveParent();
		}
		std::vector<Node*> activeChildren;
		std::vector<Node*> children = getChildren(myParent);
		for (size_t i = 0; i < children.size(); ++i) {
			if (children[i]->getType() == NodeType::RETICULATION_NODE) {
				// we need to check if the child is active, this is, if we are currently the selected parent
				if (children[i]->getReticulationData()->getLinkToActiveParent()->getTargetNode() != this) {
					continue;
				}
			}
			activeChildren.push_back(children[i]);
		}
		assert(activeChildren.size() <= 2 || (myParent == nullptr && activeChildren.size() == 3));
		return activeChildren;
	}

	std::vector<Node*> getNeighbors() const {
		std::vector<Node*> neighbors;
		for (const auto& link : links) {
			Node* target = link.getTargetNode();
			neighbors.push_back(target);
		}
		return neighbors;
	}

	std::vector<Node*> getActiveNeighbors() const {
		std::vector<Node*> activeNeighbors;
		std::vector<Node*> neighbors = getNeighbors();
		for (size_t i = 0; i < neighbors.size(); ++i) {
			if (neighbors[i]->getType() == NodeType::RETICULATION_NODE) {
				// we need to check if the neighbor is active, this is, if we are currently the selected parent
				if (neighbors[i]->getReticulationData()->getLinkToActiveParent()->getTargetNode() != this) {
					continue;
				}
			}
			activeNeighbors.push_back(neighbors[i]);
		}
		assert(activeNeighbors.size() <= 3);
		return activeNeighbors;
	}

    Node* getActiveParent() const {
        if (type == NodeType::RETICULATION_NODE) {
            return reticulationData->getActiveParent();
        }
        for (const auto& link: links) {
            if (link.direction == Direction::INCOMING) {
                return link.getTargetNode();
            }
        }
        return nullptr;
    }

        std::vector<Node*> getAllParents() const {
                std::vector<Node*> res;
                if (type == NodeType::RETICULATION_NODE) {
                        res.emplace_back(reticulationData->getFirstParent());
                        res.emplace_back(reticulationData->getSecondParent());
                } else {
                        Node* parent = getActiveParent();
                        if (parent) {
                                res.emplace_back(parent);
                        }
                }
                return res;
        }

	Edge* getEdgeTo(const Node* target) const {
		assert(target);
		for (const auto& link : links) {
			if (link.outer->node == target) {
				return link.edge;
			}
		}
		throw std::runtime_error("The given target node is not a neighbor of this node");
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

	Link* addLink(Link& link) {
		links.emplace_back(link);
		assert(links.size() <= 3);

		// update the next pointers
		for (size_t i = 0; i < links.size(); ++i) {
			links[i].next = &links[i+1 % links.size()];
		}

		return &(links[links.size() - 1]);
	}

	const std::string& getLabel() const {
		return label;
	}
	void setLabel(const std::string& label) {
		this->label = label;
	}
	NodeType getType() const {
		return type;
	}

	const std::unique_ptr<ReticulationData>& getReticulationData() const {
		assert(type == NodeType::RETICULATION_NODE);
		return reticulationData;
	}

	NodeType type = NodeType::BASIC_NODE;
	int scaler_index = -1;
	size_t clv_index = 0;
	std::vector<Link> links;
	std::unique_ptr<ReticulationData> reticulationData = nullptr;
	std::string label = "";
};

}
