/*
 * parse_unetwork_functions.c
 *
 *  Created on: Apr 26, 2019
 *      Author: sarah
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "lowlevel_parsing.h"

#define MAX_RETICULATION_COUNT 64

#ifndef NULL
#define NULL   ((void *) 0)
#endif

void * xmalloc(size_t size)
{
  void * t;
  t = malloc(size);
  if (!t)
    printf("Unable to allocate enough memory.\n");

  return t;
}

char * xstrdup(const char * s)
{
  size_t len = strlen(s);
  char * p = (char *)xmalloc(len+1);
  return strcpy(p,s);
}

static void dealloc_data(unetwork_node_t * node, void (*cb_destroy)(void *))
{
  if (node->data)
  {
    if (cb_destroy)
      cb_destroy(node->data);
  }
}

static void dealloc_graph_recursive(unetwork_node_t * node,
                                   void (*cb_destroy)(void *),
                                   int level)
{
  if (!node->next)
  {
    /* tip node */
    dealloc_data(node, cb_destroy);
    free(node->label);
    if (node->reticulation_name)
      free(node->reticulation_name);
    free(node);
  }
  else
  {
    /* inner node */
    if (node->label)
      free(node->label);
    if (node->reticulation_name)
      free(node->reticulation_name);

    unetwork_node_t * snode = node;
	do
    {
      if (node != snode || level == 0)
        dealloc_graph_recursive(snode->back, cb_destroy, level+1);
      unetwork_node_t * next = snode->next;
      dealloc_data(snode, cb_destroy);
      free(snode);
      snode = next;
    }
    while(snode && snode != node);
  }
}

void unetwork_graph_destroy(unetwork_node_t * root,
                                        void (*cb_destroy)(void *))
{
  if (!root) return;

  dealloc_graph_recursive(root, cb_destroy, 0);
}

void unetwork_destroy(unetwork_t * network,
                                  void (*cb_destroy)(void *))
{
  unsigned int i;
  unetwork_node_t * node;

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
  if (network->nodes)
  {
    free(network->nodes);
  }
  if (network->reticulation_nodes)
  {
    free(network->reticulation_nodes);
  }
  free(network);
}

static int unetwork_is_rooted(const unetwork_node_t * root)
{
  return (root->next && root->next->next == root) ? 1 : 0;
}

unsigned int count_incoming(const unetwork_node_t * node) {
	assert(node);
	unsigned int cnt = 0;
	assert(node->incoming == 0 || node->incoming == 1);
	if (node->incoming) {
		cnt++;
	}
	unetwork_node_t * snode = node->next;
	while (snode && snode != node) {
		assert(snode->incoming == 0 || snode->incoming == 1);
		if (snode->incoming) {
			cnt++;
		}
		snode = snode->next;
	}
	return cnt;
}

int unetwork_is_reticulation(const unetwork_node_t * node) {
	assert(node);
	unsigned int cnt_in = count_incoming(node);
	return (cnt_in > 1);
}

static void fill_nodes_recursive(unetwork_node_t * node,
                                         unetwork_node_t ** array,
		                                 unetwork_node_t ** reticulation_nodes,
                                         unsigned int array_size,
                                         unsigned int * tip_index,
                                         unsigned int * inner_index,
                                         unsigned int level,
	                                     int* visited_reticulations) // TODO: Is this correct?
{
  if (!node || (unetwork_is_reticulation(node) && visited_reticulations[node->reticulation_index])) {
	  return;
  }

  unetwork_node_t * snode = node;
  do
  {
	if (!snode->incoming) // outgoing edge
	{
	  fill_nodes_recursive(snode->back, array, reticulation_nodes, array_size, tip_index, inner_index, level + 1, visited_reticulations);
	}
	snode = snode->next;
  } while (snode && snode != node);

  // now, we are at the node itself.
  unsigned int index;
  if (!node->next)
  {
	index = *tip_index;
	*tip_index += 1;
  }
  else
  { // inner node
	if (unetwork_is_reticulation(node)) {
	  	reticulation_nodes[node->reticulation_index] = node;
	  	visited_reticulations[node->reticulation_index] += 1;
	}
	// TODO: We need to ensure that we only put one representative of each node into the array!!! That is, only the incoming node...
	assert(level == 0 || node->incoming);
	index = *inner_index;
	*inner_index += 1;
  }
  assert(index < array_size);
  array[index] = node;
}

static unsigned int unetwork_count_nodes_recursive(unetwork_node_t * node,
                                                unsigned int * tip_count,
                                                unsigned int * inner_tree_count,
												unsigned int * reticulation_count,
                                                unsigned int level,
												int* visited_reticulations)
{
  if (!node->next) // we have a tip node
  {
    *tip_count += 1;
    return 1;
  }
  else
  {
    unsigned int count = 0;

    unetwork_node_t * snode = level ? node->next : node;
	do
	{
	  if (!snode->incoming)
	  {
		if (!unetwork_is_reticulation(snode->back) || !visited_reticulations[snode->back->reticulation_index])
	      count += unetwork_count_nodes_recursive(snode->back, tip_count, inner_tree_count, reticulation_count, level+1, visited_reticulations);
	  }
	  snode = snode->next;
	}
	while (snode != node);

	if (unetwork_is_reticulation(node)) {
	  *reticulation_count += 1;
	  visited_reticulations[node->reticulation_index] += 1;
	} else {
      *inner_tree_count += 1;
	}

	return count + 1;
  }
}

static unsigned int unetwork_count_nodes(unetwork_node_t * root, unsigned int * tip_count,
                                      unsigned int * inner_tree_count, unsigned int * reticulation_count)
{
  unsigned int count = 0;

  if (tip_count)
    *tip_count = 0;

  if (inner_tree_count)
    *inner_tree_count = 0;

  if (reticulation_count)
	*reticulation_count = 0;

  if (!root->next && !root->back->next)
    return 0;

  if (!root->next)
    root = root->back;

  int* visited_reticulations = (int *)malloc(MAX_RETICULATION_COUNT * sizeof(int));
  int i;
  for (i = 0; i < MAX_RETICULATION_COUNT; ++i)
  {
	visited_reticulations[i] = 0;
  }
  count = unetwork_count_nodes_recursive(root, tip_count, inner_tree_count, reticulation_count, 0, visited_reticulations);
  free(visited_reticulations);

  if (tip_count && inner_tree_count && reticulation_count)
    assert(count == *tip_count + *inner_tree_count + *reticulation_count);

  return count;
}

static void unetwork_unset_indices(unetwork_t * network) {
	unsigned int node_count = network->tip_count + network->inner_tree_count + network->reticulation_count;
	unsigned int i;
	for (i = 0; i < node_count; ++i) {
		unetwork_node_t * snode = network->nodes[i];
		do {
			snode->clv_index = 0;
			snode->node_index = 0;
			snode->scaler_index = 0;
			snode->pmatrix_index = 0;
			snode = snode->next;
		} while (snode && snode != network->nodes[i]);
	}
}

void unetwork_set_indices(unetwork_t * network) {
	unsigned int node_count = network->tip_count + network->inner_tree_count + network->reticulation_count;
	unsigned int i;
	for (i = 0; i < network->tip_count; ++i) {
		network->nodes[i]->clv_index = i;
		network->nodes[i]->node_index = i;
		network->nodes[i]->pmatrix_index = i; // this index is for the edges...
		network->nodes[i]->back->pmatrix_index = i;
		network->nodes[i]->scaler_index = -1;
	}
	unsigned int clv_idx = network->tip_count;
	unsigned int node_idx = network->tip_count;
	unsigned int pmatrix_idx = network->tip_count;

	for (i = network->tip_count; i < node_count; ++i) {
		unetwork_node_t * snode = network->nodes[i];
		do {
			snode->clv_index = clv_idx;
			snode->node_index = node_idx++;
			snode->scaler_index = snode->clv_index - network->tip_count;
			if (snode->back == network->nodes[0]) {
			} else if (snode->back->pmatrix_index == 0) {
				snode->pmatrix_index = pmatrix_idx;
				snode->back->pmatrix_index = pmatrix_idx++;
				assert(snode->pmatrix_index == snode->back->pmatrix_index);
			} else {
				snode->pmatrix_index = snode->back->pmatrix_index;
			}
			snode = snode->next;
		} while (snode && snode != network->nodes[i]);
		++clv_idx;
	}

	assert(network->vroot->back == network->nodes[0] || network->vroot->pmatrix_index != 0);
	for (i = 0; i < node_count; ++i) {
		assert(network->nodes[i]->pmatrix_index == network->nodes[i]->back->pmatrix_index);
		if (i < network->tip_count) {
			assert(network->nodes[i]->pmatrix_index == network->nodes[i]->clv_index);
			assert(network->nodes[i]->back->pmatrix_index == network->nodes[i]->clv_index);
		}
	}
}

static unetwork_t * unetwork_wrapnetwork(unetwork_node_t * root,
                                    unsigned int tip_count,
                                    unsigned int inner_tree_count,
									unsigned int reticulation_count,
                                    int binary)
{
  unsigned int node_count;

  unetwork_t * network = (unetwork_t *)malloc(sizeof(unetwork_t));
  if (!network)
  {
    printf("Unable to allocate enough memory.\n");
  }

  if (tip_count < 3 && tip_count != 0)
  {
    printf("Invalid tip_count value (%u).\n", tip_count);
  }

  if (!root->next)
    root = root->back;

  if (binary)
  {
    if (tip_count == 0)
    {
      node_count = unetwork_count_nodes(root, &tip_count, &inner_tree_count, &reticulation_count);
      if (inner_tree_count != tip_count - 2 + reticulation_count)
      {
        printf("Input network is not strictly bifurcating.\n");
      }
    }
    else
    {
      inner_tree_count = tip_count - 2 + reticulation_count;
      node_count = tip_count + inner_tree_count + reticulation_count;
    }
  }
  else
  {
    if (tip_count == 0 || inner_tree_count == 0)
      node_count = unetwork_count_nodes(root, &tip_count, &inner_tree_count, &reticulation_count);
    else
      node_count = tip_count + inner_tree_count + reticulation_count;
  }

  if (!tip_count)
  {
    printf("Input network contains no inner nodes.\n");
  }

  network->nodes = (unetwork_node_t **)malloc(node_count*sizeof(unetwork_node_t *));
  if (!network->nodes)
  {
    printf("Unable to allocate enough memory.\n");
  }

  if (reticulation_count > 0)
  {
    network->reticulation_nodes = (unetwork_node_t **)malloc(reticulation_count*sizeof(unetwork_node_t *));
	if (!network->reticulation_nodes)
	{
	  printf("Unable to allocate enough memory.\n");
	}
  }
  else
  {
	network->reticulation_nodes = NULL;
  }

  unsigned int tip_index = 0;
  unsigned int inner_index = tip_count;

  int* visited_reticulations = (int *)malloc(MAX_RETICULATION_COUNT * sizeof(int));
  int i;
  for (i = 0; i < MAX_RETICULATION_COUNT; ++i)
  {
  	visited_reticulations[i] = 0;
  }
  fill_nodes_recursive(root, network->nodes, network->reticulation_nodes, node_count, &tip_index, &inner_index, 0, visited_reticulations);
  free(visited_reticulations);

  assert(tip_index == tip_count);
  assert(inner_index == tip_count + inner_tree_count + reticulation_count);

  network->tip_count = tip_count;
  network->inner_tree_count = inner_tree_count;
  network->reticulation_count = reticulation_count;
  network->edge_count = network->tip_count + network->inner_tree_count - 1 + 2 * network->reticulation_count;
  network->tree_edge_count = network->tip_count + network->inner_tree_count - 1; //network->inner_tree_count * 2 + 1; //?
  network->binary = (inner_tree_count == tip_count - (unetwork_is_rooted(root) ? 1 : 2));
  network->vroot = root;

  unetwork_unset_indices(network); // just to be sure
  unetwork_set_indices(network);

  return network;
}

unetwork_t * unetwork_wrapnetwork_multi(unetwork_node_t * root,
                                                  unsigned int tip_count,
                                                  unsigned int inner_tree_count,
												  unsigned int reticulation_count)
{
  return unetwork_wrapnetwork(root, tip_count, inner_tree_count, reticulation_count, 0);
}

/* wraps/encalupsates the unrooted network graph into a network structure
   that contains a list of nodes, number of tips and number of inner
   nodes. If 0 is passed as tip_count, then an additional recrursion
   of the network structure is done to detect the number of tips */
unetwork_t * unetwork_wrapnetwork_main(unetwork_node_t * root,
                                            unsigned int tip_count)
{
  return unetwork_wrapnetwork(root, tip_count, 0, 0, 1);
}

unetwork_node_t * unetwork_unroot_inplace(unetwork_node_t * root)
{
  /* check for a bifurcation at the root */
  if (unetwork_is_rooted(root))
  {
    unetwork_node_t * left = root->back;
    unetwork_node_t * right =  root->next->back;

    if (root->label)
      free(root->label);
    if (root->reticulation_name)
      free(root->reticulation_name);
    free(root->next);
    free(root);

    double new_length = left->length + right->length;
    left->back = right;
    right->back = left;
    left->length = right->length = new_length;
    left->prob = 1.0;

    if (left->pmatrix_index < right->pmatrix_index) {
    	right->pmatrix_index = left->pmatrix_index;
    } else {
    	left->pmatrix_index = right->pmatrix_index;
    }

    return left->next ? left : right;
  }
  else
  	return root;
}

static unetwork_node_t * rnetwork_unroot(rnetwork_node_t * root, unetwork_node_t * back, unetwork_node_t** reticulation_nodes)
{
  unetwork_node_t * uroot;
  if (!root->is_reticulation) {
	  uroot = create_unetwork_node_t();
	  if (!uroot)
	  {
	    printf("Unable to allocate enough memory.\n");
	    return NULL;
	  }
	  uroot->back = back;
	  uroot->label = (root->label) ? xstrdup(root->label) : NULL;
	  uroot->reticulation_name = NULL;
	  uroot->reticulation_index = -1;
	  uroot->length = uroot->back->length;
	  uroot->prob = uroot->back->prob;

	  if (!root->left && !root->right)
	  {
		uroot->next = NULL;
		return uroot;
	  }

	  uroot->next = create_unetwork_node_t();
	  if (!uroot->next)
	  {
		free(uroot);
		printf("Unable to allocate enough memory.\n");
		return NULL;
	  }

	  uroot->next->next = create_unetwork_node_t();
	  if (!uroot->next->next)
	  {
		free(uroot->next);
		free(uroot);
		printf("Unable to allocate enough memory.\n");
		return NULL;
	  }

	  uroot->next->next->next = uroot;

	  if (root->left->is_reticulation) {
		if (root->left->first_parent == root) {
	      uroot->next->length = root->left->first_parent_length;
		  uroot->next->prob = root->left->first_parent_prob;
		} else {
		  uroot->next->length = root->left->second_parent_length;
		  uroot->next->prob = root->left->second_parent_prob;
		}
	  } else {
		uroot->next->length = root->left->length;
		uroot->next->prob = 1.0;
	  }
	  uroot->next->label = uroot->label;
	  uroot->next->reticulation_index = uroot->reticulation_index;
	  uroot->next->reticulation_name = uroot->reticulation_name;
	  uroot->next->active = 1;
	  uroot->next->incoming = 0;
	  uroot->next->back = rnetwork_unroot(root->left, uroot->next, reticulation_nodes);
	  uroot->next->back->active = 1;
	  uroot->next->back->incoming = 1;

	  if (root->right->is_reticulation) {
	    if (root->right->first_parent == root) {
	  	  uroot->next->next->length = root->right->first_parent_length;
	  	  uroot->next->next->prob = root->right->first_parent_prob;
	    } else {
	  	  uroot->next->next->length = root->right->second_parent_length;
	  	  uroot->next->next->prob = root->right->second_parent_prob;
	    }
	  } else {
	  	uroot->next->next->length = root->right->length;
	  	uroot->next->next->prob = 1.0;
	  }
	  uroot->next->next->label = uroot->label;
	  uroot->next->next->reticulation_index = uroot->reticulation_index;
	  uroot->next->next->reticulation_name = uroot->reticulation_name;

	  uroot->next->next->active = 1;
	  uroot->next->next->incoming = 0;
	  uroot->next->next->back = rnetwork_unroot(root->right, uroot->next->next, reticulation_nodes);
	  uroot->next->next->back->active = 1;
	  uroot->next->next->back->incoming = 1;
  } else { // now, we have a reticulation node
	// first, check if we already created a node for this reticulation
	if (!reticulation_nodes[root->reticulation_index]) {
      uroot = create_unetwork_node_t();
	  if (!uroot)
	  {
		printf("Unable to allocate enough memory.\n");
		return NULL;
	  }
	  uroot->back = back;
	  uroot->label = (root->label) ? xstrdup(root->label) : NULL;
	  uroot->reticulation_name = (root->reticulation_name) ? xstrdup(root->reticulation_name) : NULL;
	  uroot->reticulation_index = root->reticulation_index;

	  uroot->length = uroot->back->length;
	  uroot->prob = uroot->back->prob;

	  if (!root->child)
	  {
		free(uroot);
		printf("The reticulation node has no child.\n");
		return NULL;
	  }

	  uroot->next = create_unetwork_node_t();
	  if (!uroot->next)
	  {
        free(uroot);
		printf("Unable to allocate enough memory.\n");
		return NULL;
	  }

	  uroot->next->next = create_unetwork_node_t();
	  if (!uroot->next->next)
	  {
	    free(uroot->next);
		free(uroot);
		printf("Unable to allocate enough memory.\n");
		return NULL;
	  }

	  reticulation_nodes[root->reticulation_index] = uroot->next->next;

	  uroot->next->next->next = uroot;

	  if (root->child->is_reticulation) {
	    if (root->child->first_parent == root) {
	      uroot->next->length = root->child->first_parent_length;
		  uroot->next->prob = root->child->first_parent_prob;
	    } else {
		  uroot->next->length = root->child->second_parent_length;
		  uroot->next->prob = 1.0 - root->child->second_parent_prob;
		}
	  } else {
	    uroot->next->length = root->child->length;
		uroot->next->prob = 1.0;
	  }
	  uroot->next->label = uroot->label;
	  uroot->next->reticulation_name = uroot->reticulation_name;
	  uroot->next->reticulation_index = uroot->reticulation_index;

	  uroot->next->active = 1;
      uroot->next->incoming = 0;
	  uroot->next->back = rnetwork_unroot(root->child, uroot->next, reticulation_nodes);
	  uroot->next->back->active = 1;
	  uroot->next->back->incoming = 1;

	  uroot->next->next->label = uroot->label;
	  uroot->next->next->reticulation_name = uroot->reticulation_name;
	  uroot->next->next->reticulation_index = uroot->reticulation_index;
	} else {
	  uroot = reticulation_nodes[root->reticulation_index];
	  uroot->back = back;
	  uroot->prob = uroot->back->prob;
	  uroot->length = uroot->back->length;
	}
  }

  return uroot;
}

unetwork_t * rnetwork_unroot_main(rnetwork_t * network) {
  // for each node in the rnetwork, we need to create three nodes in the unetwork... except for the leaves.
  unetwork_node_t** reticulation_nodes = (unetwork_node_t**)malloc(network->reticulation_count * sizeof(unetwork_node_t*)); // we only need one representative per reticulation node
  // ... we should explicitly fill the reticulation nodes array with null pointers.
  for (unsigned int i = 0; i < network->reticulation_count; ++i) {
	  reticulation_nodes[i] = NULL;
  }

  rnetwork_node_t * root = network->root;

  if (!root->left->left && !root->right->left)
  {
	printf("Network requires at least three tips to be converted to unrooted\n");
	return NULL;
  }

  rnetwork_node_t * new_root;

  unetwork_node_t * uroot = create_unetwork_node_t();
  if (!uroot)
  {
	printf("Unable to allocate enough memory.\n");
	return NULL;
  }

  uroot->next = create_unetwork_node_t();
  if (!uroot->next)
  {
	free(uroot);
	printf("Unable to allocate enough memory.\n");
	return NULL;
  }

  uroot->next->next = create_unetwork_node_t();
  if (!uroot->next->next)
  {
	free(uroot->next);
	free(uroot);
	printf("Unable to allocate enough memory.\n");
	return NULL;
  }

  uroot->next->next->next = uroot;

  uroot->length = root->left->length + root->right->length;
  uroot->prob = 1.0;

  /* get the first root child that has descendants and make it the new root */
  if (root->left->left)
  {
	new_root = root->left;
	uroot->back = rnetwork_unroot(root->right,uroot, reticulation_nodes);
	/* TODO: Need to clean uroot in case of error */
	if (!uroot->back) return NULL;
  }
  else
  {
	new_root = root->right;
	uroot->back = rnetwork_unroot(root->left,uroot, reticulation_nodes);
	/* TODO: Need to clean uroot in case of error*/
	if (!uroot->back) return NULL;
  }

  uroot->label = (new_root->label) ? xstrdup(new_root->label) : NULL;
  uroot->reticulation_name = NULL;
  uroot->active = 1;
  uroot->incoming = 0;
  uroot->back->active = 1;
  uroot->back->incoming = 1;
  uroot->reticulation_index = -1;

  uroot->next->active = 1;
  uroot->next->incoming = 0;
  uroot->next->label = uroot->label;
  uroot->next->reticulation_name = uroot->reticulation_name;
  uroot->next->reticulation_index = -1;
  if (new_root->left->is_reticulation)
  {
    if (new_root->left->first_parent == new_root)
    {
      uroot->next->prob = new_root->left->first_parent_prob;
      uroot->next->length = new_root->left->first_parent_length;
    } else
    {
      uroot->next->prob = new_root->left->second_parent_prob;
      uroot->next->length = new_root->left->second_parent_length;
    }
  }
  else
  {
  	uroot->next->length = new_root->left->length;
    uroot->next->prob = 1.0;
  }
  uroot->next->back = rnetwork_unroot(new_root->left, uroot->next, reticulation_nodes);
  /* TODO: Need to clean uroot in case of error*/
  if (!uroot->next->back) return NULL;
  uroot->next->back->active = 1;
  uroot->next->back->incoming = 1;

  uroot->next->next->active = 1;
  uroot->next->next->incoming = 0;
  uroot->next->next->label = uroot->label;
  uroot->next->next->reticulation_name = uroot->reticulation_name;
  uroot->next->next->reticulation_index = -1;

  if (new_root->right->is_reticulation)
  {
    if (new_root->right->first_parent == new_root)
    {
      uroot->next->next->prob = new_root->right->first_parent_prob;
      uroot->next->next->length = new_root->right->first_parent_length;
    } else
    {
      uroot->next->next->prob = new_root->right->second_parent_prob;
      uroot->next->next->length = new_root->right->second_parent_length;
    }
  }
  else
  {
	uroot->next->next->length = new_root->right->length;
	uroot->next->next->prob = 1.0;
  }
  uroot->next->next->back = rnetwork_unroot(new_root->right, uroot->next->next, reticulation_nodes);
  /* TODO: Need to clean uroot in case of error*/
  if (!uroot->next->next->back) return NULL;
  uroot->next->next->back->active = 1;
  uroot->next->next->back->incoming = 1;

  free(reticulation_nodes);
  unetwork_t * res = unetwork_wrapnetwork_main(uroot,0);
  // re-wire pointer to fit the utree conventions...
  res->vroot = res->vroot->next;
  res->nodes[res->vroot->clv_index] = res->vroot;
  return res;
}

unetwork_t * unetwork_parse_newick_string(const char * s)
{
  rnetwork_t * rnetwork = rnetwork_parse_newick_string(s);
  assert(rnetwork);
  unetwork_t * unetwork = rnetwork_unroot_main(rnetwork);
  unetwork->binary = rnetwork->binary;
  rnetwork_destroy(rnetwork, NULL);
  return unetwork;
}

unetwork_t * unetwork_parse_newick(const char * filename)
{
  rnetwork_t * rnetwork = rnetwork_parse_newick(filename);
  unetwork_t * unetwork = rnetwork_unroot_main(rnetwork);
  rnetwork_destroy(rnetwork, NULL);
  return unetwork;
}

unetwork_node_t * create_unetwork_node_t() {
	unetwork_node_t* node = (unetwork_node_t*) malloc(sizeof(unetwork_node_t));
	node->length = 0.0;
	node->prob = 1.0;
	node->support = 0.0;
	node->node_index = 0;
	node->clv_index = 0;
	node->scaler_index = -1;
	node->pmatrix_index = 0;
	node->reticulation_index = -1;
	node->next = NULL;
	node->back = NULL;
	node->incoming = 0;
	node->active = 1;
	node->label = NULL;
	node->reticulation_name = NULL;
	node->data = NULL;
	return node;
}


