/*
 * parse_rnetwork_functions.c
 *
 *  Created on: Apr 26, 2019
 *      Author: sarah
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "lowlevel_parsing.h"

#ifndef NULL
#define NULL   ((void *) 0)
#endif

static void fill_nodes_recursive(rnetwork_node_t * node,
                                 rnetwork_node_t ** array,
                                 rnetwork_node_t ** reticulations,
                                 unsigned int * tip_index,
                                 unsigned int * inner_index,
                                 unsigned int * scaler_index)
{
  if (!node)
  {
    return;
  }

  if (!node->is_reticulation && !node->left && !node->right)
  { // we are at a tip node
    node->idx = *tip_index;
    node->clv_index = *tip_index;
    node->pmatrix_index = *tip_index;
    node->scaler_index = -1;
    array[*tip_index] = node;
    *tip_index = *tip_index + 1;
    return;
  }

  if (node->idx > 0) // the node has already been visited
  {
    return;
  }

  if (!node->is_reticulation)
  {
    fill_nodes_recursive(node->left,  array, reticulations, tip_index, inner_index, scaler_index);
    fill_nodes_recursive(node->right, array, reticulations, tip_index, inner_index, scaler_index);
  }
  else
  {
    reticulations[node->reticulation_index] = node;
    fill_nodes_recursive(node->child, array, reticulations, tip_index, inner_index, scaler_index);
  }

  array[*inner_index] = node;
  node->idx = *inner_index;
  node->clv_index = *inner_index;
  node->pmatrix_index = *inner_index;
  node->scaler_index = *scaler_index;
  *inner_index = *inner_index + 1;
  *scaler_index = *scaler_index + 1;
}

static unsigned int rnetwork_count_nodes_recursive(rnetwork_node_t* parent, rnetwork_node_t * node,
                                                unsigned int * tip_count,
                                                unsigned int * inner_tree_count,
												unsigned int * reticulation_count)
{
  if (!node)
  {
	return 0;
  }

  if (!node->is_reticulation)
  {
	 if (!node->left && !node->right)
	 {
	   *tip_count += 1;
	   return 1;
	 }
	 else
	 {
	   *inner_tree_count += 1;
	   return 1 + rnetwork_count_nodes_recursive(node, node->left, tip_count, inner_tree_count, reticulation_count) + rnetwork_count_nodes_recursive(node, node->right, tip_count, inner_tree_count, reticulation_count);
	 }
  }
  else if (node->first_parent == parent) // don't visit reticulations more than once
  {
	*reticulation_count += 1;
	return 1 + rnetwork_count_nodes_recursive(node, node->child, tip_count, inner_tree_count, reticulation_count);
  } else {
	  return 0;
  }
}

static unsigned int rnetwork_count_nodes(rnetwork_node_t * root, unsigned int * tip_count,
                                      unsigned int * inner_tree_count, unsigned int * reticulation_count)
{
  unsigned int count = 0;

  if (tip_count)
    *tip_count = 0;

  if (inner_tree_count)
    *inner_tree_count = 0;

  if (reticulation_count)
	*reticulation_count = 0;

  if (!root)
	return 0;

  count = rnetwork_count_nodes_recursive(NULL, root, tip_count, inner_tree_count, reticulation_count);

  if (tip_count && inner_tree_count && reticulation_count)
    assert(count == *tip_count + *inner_tree_count + *reticulation_count);

  return count;
}

rnetwork_t * rnetwork_wrapnetwork(rnetwork_node_t * root)
{
  rnetwork_t * network = (rnetwork_t *)malloc(sizeof(rnetwork_t));
  if (!network)
  {
    printf("Unable to allocate enough memory.\n");
    return 0;
  }

  unsigned int tip_count = 0;
  unsigned int inner_tree_count = 0;
  unsigned int reticulation_count = 0;
  unsigned int node_count = rnetwork_count_nodes(root, &tip_count, &inner_tree_count, &reticulation_count);
  network->tip_count = tip_count;
  network->inner_tree_count = inner_tree_count;
  network->reticulation_count = reticulation_count;

  network->nodes = (rnetwork_node_t **)malloc(node_count * sizeof(rnetwork_node_t *));
  network->reticulation_nodes = (rnetwork_node_t **)malloc(reticulation_count * sizeof(rnetwork_node_t *));
  if (!network->nodes || !network->reticulation_nodes)
  {
    printf("Unable to allocate enough memory.\n");
    return 0;
  }

  unsigned int tip_index = 0;
  unsigned int inner_index = tip_count;
  unsigned int scaler_index = 0;

  fill_nodes_recursive(root->left, network->nodes, network->reticulation_nodes, &tip_index, &inner_index, &scaler_index);
  fill_nodes_recursive(root->right, network->nodes, network->reticulation_nodes, &tip_index, &inner_index, &scaler_index);

  root->idx = inner_index;
  root->clv_index = inner_index;
  root->pmatrix_index = inner_index;
  root->scaler_index = scaler_index;
  network->nodes[inner_index] = root;
  network->root = root;
  network->edge_count = reticulation_count + inner_tree_count * 2;
  network->tree_edge_count = inner_tree_count * 2;
  network->binary = 1;

  // just for debug: print the node labels
  unsigned int i;
  printf("Just for debug: print the node labels\n");
  for (i = 0; i < node_count; ++i) {
	  assert(network->nodes[i]);
	  if (network->nodes[i]->label) {
		  printf("%d: %s\n", i, network->nodes[i]->label);
	  } else {
		  printf("%d: NULL\n", i);
	  }
  }
  unsigned int j;
  // just for debug: ensure that no two node pointers are the same
  for (i = 0; i < node_count; ++i) {
	  for (j = i + 1; j < node_count; ++j) {
		  assert(network->nodes[i] != network->nodes[j]);
	  }
  }
  // just for debug: ensure that the retcículation name is NULL for non-reticulation nodes
  for (i = 0; i < node_count; ++i) {
	  assert((network->nodes[i]->is_reticulation == 1 && network->nodes[i]->reticulation_name != NULL) || (network->nodes[i]->is_reticulation == 0 && network->nodes[i]->reticulation_name == NULL));
  }

  return network;
}

static void dealloc_data(rnetwork_node_t * node, void (*cb_destroy)(void *))
{
  if (node->data)
  {
    if (cb_destroy)
      cb_destroy(node->data);
  }
}

void rnetwork_graph_destroy(rnetwork_node_t * root,
                                        void (*cb_destroy)(void *))
{
  if (!root) return;

  rnetwork_graph_destroy(root->left, cb_destroy);
  rnetwork_graph_destroy(root->right, cb_destroy);

  dealloc_data(root, cb_destroy);
  free(root->label);
  free(root);
}

void rnetwork_destroy(rnetwork_t * network,
                                  void (*cb_destroy)(void *))
{
  unsigned int i;
  rnetwork_node_t * node;


  // just for debug: ensure that no two node pointers are the same
  unsigned int node_count = network->tip_count + network->inner_tree_count + network->reticulation_count;
  unsigned int j;
  for (i = 0; i < node_count; ++i) {
	  for (j = i + 1; j < node_count; ++j) {
		  assert(network->nodes[i] != network->nodes[j]);
	  }
  }
  // just for debug: ensure that the reticulation name is NULL for non-reticulation nodes
    for (i = 0; i < node_count; ++i) {
  	  assert((network->nodes[i]->is_reticulation == 1 && network->nodes[i]->reticulation_name != NULL) || (network->nodes[i]->is_reticulation == 0 && network->nodes[i]->reticulation_name == NULL));
    }


  /* deallocate all nodes */
  for (i = 0; i < network->tip_count + network->inner_tree_count + network->reticulation_count; ++i)
  {
    node = network->nodes[i];
    dealloc_data(node, cb_destroy);

    if (node->label)
      free(node->label);
    if (node->reticulation_name)
      free(node->reticulation_name);

    free(node);
  }

  /* deallocate network structure */
  free(network->nodes);
  free(network->reticulation_nodes);
  free(network);
}

rnetwork_node_t * create_rnetwork_node_t() {
	rnetwork_node_t* node = (rnetwork_node_t*) malloc(sizeof(rnetwork_node_t));

	node->label = NULL;
	node->idx = 0;
	node->clv_index = 0;
	node->pmatrix_index = 0;
	node->is_reticulation = 0;
	node->data = NULL;
	node->scaler_index = -1;

	node->length = 0.0; // length of the edge to the parent node
	node->parent = NULL; // in case of a reticulation node, this can be either first_parent, right_parent, or NULL (undecided).

		// the following fields are only relevant if it is a reticulation node
	node->reticulation_name = NULL;
	node->reticulation_index = -1; // -1 if not a reticulation
	node->support = 0.0;
	node->first_parent = NULL; // the first parent has to be a non-reticulation node
	node->first_parent_length = 0.0; // length of the edge from the first parent node
	node->first_parent_prob = 0.5;
	node->second_parent_prob = 0.5;
	node->second_parent_length = 0.0; // length of the edge from the second parent node
	node->second_parent = NULL; // the second parent has to be a non-reticulation node
	node->child = NULL; // the child has to be a non-reticulation node

		// the following fields are only relevant if it is not a reticulation node
	node->left = NULL;
	node->right = NULL;

	return node;
}
