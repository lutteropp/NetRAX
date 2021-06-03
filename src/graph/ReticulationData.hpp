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

namespace netrax {

class ReticulationData {
 public:
  void init(size_t index, const std::string &label, bool activeParent,
            Link *linkToFirstParent, Link *linkToSecondParent,
            Link *linkToChild) {
    this->reticulation_index = index;
    this->label = label;
    this->active_parent_toggle = activeParent;
    this->link_to_first_parent = linkToFirstParent;
    this->link_to_second_parent = linkToSecondParent;
    this->link_to_child = linkToChild;
  }

  size_t getReticulationIndex() const { return reticulation_index; }
  const std::string &getLabel() const { return label; }
  Link *getLinkToActiveParent() const {
    if (active_parent_toggle == 0) {
      return link_to_first_parent;
    } else {
      return link_to_second_parent;
    }
  }

  Link *getLinkToNonActiveParent() const {
    if (active_parent_toggle == 1) {
      return link_to_first_parent;
    } else {
      return link_to_second_parent;
    }
  }

  void setActiveParentToggle(bool val) { active_parent_toggle = val; }
  Link *getLinkToFirstParent() const { return link_to_first_parent; }
  Link *getLinkToSecondParent() const { return link_to_second_parent; }
  Link *getLinkToChild() const { return link_to_child; }
  void setLinkToFirstParent(Link *link) { link_to_first_parent = link; }
  void setLinkToSecondParent(Link *link) { link_to_second_parent = link; }
  void setLinkToChild(Link *link) { link_to_child = link; }

  bool active_parent_toggle = 0;  // 0: first_parent, 1: second_parent
  size_t reticulation_index = 0;
  Link *link_to_first_parent =
      nullptr;  // The link that has link->outer->node as the first parent
  Link *link_to_second_parent =
      nullptr;  // The link that has link->outer->node as the second parent
  Link *link_to_child =
      nullptr;  // The link that has link->outer->node as the child
  std::string label = "";
};

}  // namespace netrax
