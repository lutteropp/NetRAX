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
	void initBasic(size_t index, int scaler_index, Link* link, const std::string& label) {
		this->clv_index = index;
		this->scaler_index = scaler_index;
		this->link = link;
		this->label = label;
		this->type = NodeType::BASIC_NODE;
		this->reticulationData = nullptr;
	}
	void initReticulation(size_t index, int scaler_index, Link* link, const std::string& label, const ReticulationData& retData) {
		this->clv_index = index;
		this->scaler_index = scaler_index;
		this->link = link;
		this->label = label;
		this->type = NodeType::RETICULATION_NODE;
		reticulationData = std::make_unique<ReticulationData>(retData);
	}

	bool isTip() const {
		assert(this->link);
		return (!this->link->next);
	}

	std::vector<Node*> getChildren(const Node* myParent) const {
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

	std::vector<Node*> getActiveChildren(const Node* myParent) const {
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
		Link* currLink = link;
		do {
			Node* target = currLink->getTargetNode();
			if (std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
				throw std::runtime_error("Loop in neighbors list!");
			}
			neighbors.push_back(target);
			if (!currLink->next) { // leaf node
				assert(neighbors.size() == 1);
				break;
			}
			currLink = currLink->next;
		} while (currLink != link);
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

	Edge* getEdgeTo(const Node* target) const {
		assert(target);
		Link* currLink = link;
		while (currLink != nullptr) {
			if (currLink->outer->node == target) {
				return currLink->edge;
			}
			currLink = currLink->next;
			if (currLink == link) {
				break;
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

	Link* getLink() const {
		return link;
	}
	void setLink(Link* link) {
		this->link = link;
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
	Link* link = nullptr;
	std::unique_ptr<ReticulationData> reticulationData = nullptr;
	std::string label = "";
};

}
