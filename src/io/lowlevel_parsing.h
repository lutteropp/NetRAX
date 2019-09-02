/*
 * parsing.h
 *
 *  Created on: Sep 2, 2019
 *      Author: Sarah Lutteropp
 */

#pragma once

typedef struct rnetwork_node_s
{
  char* label;
  unsigned int idx; // index in the nodes array
  unsigned int clv_index; // same as idx
  unsigned int pmatrix_index; // same as idx
  int is_reticulation;
  void* data;
  int scaler_index;

  double length; // length of the edge to the parent node
  struct rnetwork_node_s* parent; // in case of a reticulation node, this can be either first_parent, right_parent, or NULL (undecided).

  // the following fields are only relevant if it is a reticulation node
  char* reticulation_name;
  int reticulation_index; // -1 if not a reticulation
  double support;
  struct rnetwork_node_s* first_parent; // the first parent has to be a non-reticulation node
  double first_parent_length; // length of the edge from the first parent node
  double first_parent_prob;
  double second_parent_prob;
  double second_parent_length; // length of the edge from the second parent node
  struct rnetwork_node_s* second_parent; // the second parent has to be a non-reticulation node
  struct rnetwork_node_s* child; // the child has to be a non-reticulation node

  // the following fields are only relevant if it is not a reticulation node
  struct rnetwork_node_s* left;
  struct rnetwork_node_s* right;
} rnetwork_node_t;

typedef struct rnetwork_s
{
  unsigned int tip_count;
  unsigned int inner_tree_count;
  unsigned int reticulation_count;
  unsigned int edge_count;
  unsigned int tree_edge_count;
  int binary;
  rnetwork_node_t ** nodes; // pointers to all nodes in the network, both tree nodes and reticulation nodes
  rnetwork_node_t ** reticulation_nodes; // pointers to all reticulation nodes in the network
  rnetwork_node_t * root;
} rnetwork_t;

typedef struct unetwork_node_s
{
  double length;
  double prob;
  double support;
  unsigned int node_index; // unique index for just this subnode / link. In a binary network, a node consists of three subnodes.
  unsigned int clv_index;
  int scaler_index;
  unsigned int pmatrix_index;
  int reticulation_index; // -1 if it is not a reticulation
  struct unetwork_node_s * next;
  struct unetwork_node_s * back;
  int incoming; // 1 for incoming, 0 for outgoing edge
  int active; // determines whether the connection is active or not, this is needed for working with an induced tree

  // we have a reticulation, if two out of node, node->next, node->next->next are incoming and one is outgoing.
  // we have an inner tree node, if two out of node, node->next, node->next->next are outgoing and one is incoming.
  // we have the root node, if all of node, node->next, and node->next->next are outgoing.
  // we have a leaf node, if node->next and node->next->next are NULL.
  char* label;
  char* reticulation_name;

  void * data;
} unetwork_node_t;

typedef struct unetwork_s
{
  unsigned int tip_count;
  unsigned int inner_tree_count;
  unsigned int reticulation_count;
  unsigned int edge_count;
  unsigned int tree_edge_count;
  int binary;

  unetwork_node_t ** nodes; // pointers to all nodes in the network, both tree nodes and reticulation nodes
  unetwork_node_t ** reticulation_nodes; // pointers to all reticulation nodes in the network
  unetwork_node_t * vroot;
} unetwork_t;



/* functions in parse_rnetwork.y */

rnetwork_t * rnetwork_parse_newick(const char * filename);

rnetwork_t * rnetwork_parse_newick_string(const char * s);

void rnetwork_destroy(rnetwork_t * root, void (*cb_destroy)(void *));

void rnetwork_graph_destroy(rnetwork_node_t * root, void (*cb_destroy)(void *));

rnetwork_t * rnetwork_wrapnetwork(rnetwork_node_t * root);

/* functions in parse_unetwork_functions.c */

unetwork_t * unetwork_parse_newick(const char * filename);

unetwork_t * unetwork_parse_newick_string(const char * s);

void unetwork_destroy(unetwork_t * root, void (*cb_destroy)(void *));

void unetwork_reset_template_indices(unetwork_node_t * node, unsigned int tip_count);

void unetwork_graph_destroy(unetwork_node_t * root, void (*cb_destroy)(void *));

void unetwork_set_indices(unetwork_t * network);

unetwork_t * unetwork_wrapnetwork(unetwork_node_t * root, unsigned int tip_count);

unetwork_t * unetwork_wrapnetwork_multi(unetwork_node_t * root, unsigned int tip_count, unsigned int inner_tree_count,
		unsigned int reticulation_count);

/* functions in parse_rnetwork_functions.c */

void rnetwork_destroy(rnetwork_t * root, void (*cb_destroy)(void *));

void rnetwork_graph_destroy(rnetwork_node_t * root, void (*cb_destroy)(void *));

rnetwork_t * rnetwork_wrapnetwork(rnetwork_node_t * root);

rnetwork_t * rnetwork_wrapnetwork_multi(rnetwork_node_t * root, unsigned int tip_count, unsigned int inner_tree_count,
		unsigned int reticulation_count);
