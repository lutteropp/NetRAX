/*
    Copyright (C) 2015 Tomas Flouri, with changes and additions by Sarah Lutteropp in 2019

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact: Tomas Flouri <Tomas.Flouri@h-its.org>,
    Heidelberg Institute for Theoretical Studies,
    Schloss-Wolfsbrunnenweg 35, D-69118 Heidelberg, Germany
*/
%{
#include "lowlevel_parsing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

extern int rnetwork_lex();
extern FILE * rnetwork_in;
extern void rnetwork_lex_destroy();
extern int rnetwork_lineno;
extern int rnetwork_colstart;
extern int rnetwork_colend;

extern int rnetwork_parse();
extern struct rnetwork_buffer_state * rnetwork__scan_string(const char * str);
extern void rnetwork__delete_buffer(struct rnetwork_buffer_state * buffer);

static unsigned int tip_cnt = 0;
static unsigned int inner_tree_cnt = 0;
static unsigned int reticulation_cnt = 0;
static rnetwork_node_t ** reticulation_node_pointers;
static char* reticulation_node_names[64];

static void rnetwork_error(rnetwork_node_t * node, const char * s)
{
  if (rnetwork_colstart == rnetwork_colend)
    printf("%s. (line %d column %d)\n",
             s, rnetwork_lineno, rnetwork_colstart);
  else
    printf("%s. (line %d column %d-%d)\n",
             s, rnetwork_lineno, rnetwork_colstart, rnetwork_colend);
}

void set_parent(rnetwork_node_t* parent, rnetwork_node_t* child) {
  assert(child);
  if (child->is_reticulation) {
    if (!child->first_parent) {
      child->first_parent = parent;
    } else {
      child->second_parent = parent;
    }
  } else {
    child->parent = parent;
  }
}

void toplevel_bifurcation(rnetwork_node_t* root, rnetwork_node_t* left, rnetwork_node_t* right, char* label, double length) {
  assert(root);
  inner_tree_cnt++;
  root->left   = left;
  root->right  = right;
  root->label  = label;
  root->length = length;
  set_parent(root, root->left);
  set_parent(root, root->right);
}

void toplevel_trifurcation(rnetwork_node_t* root, rnetwork_node_t* child1, rnetwork_node_t* child2, rnetwork_node_t* child3, char* label, double length) {
  // unrooted tree/ network -> root it!
  assert(root);
  inner_tree_cnt++;
  root->label = label;
  root->length = length;
  
  rnetwork_node_t * intermediary_node = create_rnetwork_node_t();
  inner_tree_cnt++;
  intermediary_node->left = child1;
  intermediary_node->right = child2;
  set_parent(intermediary_node, intermediary_node->left);
  set_parent(intermediary_node, intermediary_node->right);
  
  root->left = intermediary_node;
  root->right = child3;
  set_parent(root, root->left);
  set_parent(root, root->right);
}

rnetwork_node_t* inner_tree_node(rnetwork_node_t* left, rnetwork_node_t* right, char* label, double length) {
  inner_tree_cnt++;
  rnetwork_node_t* node = create_rnetwork_node_t();
  node->left = left;
  node->right = right;
  node->label = label;
  node->length = length;
  set_parent(node, node->left);
  set_parent(node, node->right);
  return node;
}

rnetwork_node_t* reticulation_node_first(rnetwork_node_t* child, char* label, char* reticulation_name, double length, double support, double probability) {
  rnetwork_node_t* node = create_rnetwork_node_t();
  node->is_reticulation = 1;
  node->child = child;
  node->label = label;
  assert(reticulation_name);
  node->reticulation_name = reticulation_name;
  node->support = support;
  node->length = length;
  node->first_parent_prob = probability;
  node->reticulation_index = reticulation_cnt;
  set_parent(node, node->child);
  reticulation_node_pointers[reticulation_cnt] = node;
  reticulation_node_names[reticulation_cnt] = node->reticulation_name;
  reticulation_cnt++;
  return node;
}

rnetwork_node_t* reticulation_node_second(char* reticulation_name, double length, double support, double probability) {
  rnetwork_node_t* node;
  unsigned int i = 0;
  for (i = 0; i < reticulation_cnt; ++i)
  {
    if (strcmp(reticulation_node_names[i], reticulation_name) == 0)
    {
      node = reticulation_node_pointers[i];
      node->second_parent_length = length;
      node->support = support;
      if (probability != 0.5) {
        node->second_parent_prob = probability;
      } else {
        node->second_parent_prob = 1.0 - node->first_parent_prob;
      }
      return node;
    }
  }
  rnetwork_error(node, "Reticulation node not found");
}

rnetwork_node_t* tip_node(char* label, double length) {
  rnetwork_node_t* node = create_rnetwork_node_t();
  node->label = label;
  node->length = length;
  tip_cnt++;
  return node;
}

%}

%union
{
  char * s;
  char * d;
  struct rnetwork_node_s * network;
}

%error-verbose
%parse-param {struct rnetwork_node_s * network}
%destructor { rnetwork_graph_destroy($$,NULL); } subnetwork
%destructor { free($$); } STRING
%destructor { free($$); } NUMBER
%destructor { free($$); } label

%token<s> STRING
%token<d> NUMBER
%type<s> label
%type<d> number
%type<network> subnetwork
%start input
%%

label: STRING    {$$=$1;} | NUMBER {$$=$1;};
number: NUMBER   {$$=$1;};

input: '(' subnetwork ',' subnetwork ')' label ':' number ';' // label + branch length
{
  toplevel_bifurcation(network, $2, $4, $6, atof($8));
  free($8);
}      | '(' subnetwork ',' subnetwork ')' label ';' // label only
{
  toplevel_bifurcation(network, $2, $4, $6, 0.0);
}      | '(' subnetwork ',' subnetwork ')' ':' number ';' // branch length only
{
  toplevel_bifurcation(network, $2, $4, NULL, atof($7));
  free($7);
}      | '(' subnetwork ',' subnetwork ')' ';' // no extra
{
  toplevel_bifurcation(network, $2, $4, NULL, 0.0);
}
       | '(' subnetwork ',' subnetwork ',' subnetwork ')' label ':' number ';' // label + branch length
{
  toplevel_trifurcation(network, $2, $4, $6, $8, atof($10));
  free($10);
}
       | '(' subnetwork ',' subnetwork ',' subnetwork ')' label ';' // label only
{
  toplevel_trifurcation(network, $2, $4, $6, $8, 0.0);
}
       | '(' subnetwork ',' subnetwork ',' subnetwork ')' ':' number ';' // branch length only
{
  toplevel_trifurcation(network, $2, $4, $6, NULL, atof($9));
  free($9);
}

       | '(' subnetwork ',' subnetwork ',' subnetwork ')' ';' // no extra
{
  toplevel_trifurcation(network, $2, $4, $6, NULL, 0.0);
}
;


subnetwork: '(' subnetwork ',' subnetwork ')' label ':' number // label + branch length
{
  $$ = inner_tree_node($2, $4, $6, atof($8));
  free($8);
}
       | '(' subnetwork ',' subnetwork ')' label // label only
{
  $$ = inner_tree_node($2, $4, $6, 0.0);
}
       | '(' subnetwork ',' subnetwork ')' ':' number // branch length only
{
  $$ = inner_tree_node($2, $4, NULL, atof($7));
  free($7);
}       
         | '(' subnetwork ',' subnetwork ')' // no extra
{
  $$ = inner_tree_node($2, $4, NULL, 0.0);
}

       | '(' subnetwork ')' label '#' label ':' number ':' number ':' number // X#H1:br_length:support:prob
{
  $$ = reticulation_node_first($2, $4, $6, atof($8), atof($10), atof($12));
  free($12);
  free($10);
  free($8);
}      
        | '(' subnetwork ')' label '#' label ':' number ':' number // X#H1:br_length:support
{
  $$ = reticulation_node_first($2, $4, $6, atof($8), atof($10), 0.5);
  free($8);
  free($10);
}
       | '(' subnetwork ')' label '#' label ':' number ':' ':' number // X#H1:br_length::prob
{
  $$ = reticulation_node_first($2, $4, $6, atof($8), 0.0, atof($11));
  free($8);
  free($11);
}
      | '(' subnetwork ')' label '#' label ':' ':' number ':' number // X#H1::support:prob
{
  $$ = reticulation_node_first($2, $4, $6, 0.0, atof($9), atof($11));
  free($9);
  free($11);
}
         | '(' subnetwork ')' label '#' label ':' number // X#H1:br_length
{
  $$ = reticulation_node_first($2, $4, $6, atof($8), 0.0, 0.5);
  free($8);
}
       | '(' subnetwork ')' label '#' label ':' ':' ':' number // X#H1:::prob
{
  $$ = reticulation_node_first($2, $4, $6, 0.0, 0.0, atof($10));
  free($10);
} 
		| '(' subnetwork ')' label '#' label ':' ':' number // X#H1::support
{
  $$ = reticulation_node_first($2, $4, $6, 0.0, atof($9), 0.5);
  free($9);
}      
         | '(' subnetwork ')' label '#' label // X#H1
{
  $$ = reticulation_node_first($2, $4, $6, 0.0, 0.0, 0.5);
}

       | '(' subnetwork ')' '#' label ':' number ':' number ':' number // #H1:br_length:support:prob
{
  $$ = reticulation_node_first($2, NULL, $5, atof($7), atof($9), atof($11));
  free($9);
  free($11);
}      
        | '(' subnetwork ')' '#' label ':' number ':' number // #H1:br_length:support
{
  $$ = reticulation_node_first($2, NULL, $5, atof($7), atof($9), 0.5);
  free($7);
  free($9);
}
      | '(' subnetwork ')' '#' label ':' ':' number ':' number // #H1::support:prob
{
  $$ = reticulation_node_first($2, NULL, $5, 0.0, atof($8), atof($10));
  free($8);
  free($10);
}      
       | '(' subnetwork ')' '#' label ':' number ':' ':' number // #H1:br_length::prob
{
  $$ = reticulation_node_first($2, NULL, $5, atof($7), 0.0, atof($10));
  free($7);
  free($10);
}
         | '(' subnetwork ')' '#' label ':' number // #H1:br_length
{
  $$ = reticulation_node_first($2, NULL, $5, atof($7), 0.0, 0.5);
  free($7);
}
       | '(' subnetwork ')' '#' label ':' ':' ':' number // #H1:::prob
{
  $$ = reticulation_node_first($2, NULL, $5, 0.0, 0.0, atof($9));
  free($9);
} 
		| '(' subnetwork ')' '#' label ':' ':' number // #H1::support
{
  $$ = reticulation_node_first($2, NULL, $5, 0.0, atof($8), 0.5);
  free($8);
}
        | '(' subnetwork ')' '#' label // #H1
{
  $$ = reticulation_node_first($2, NULL, $5, 0.0, 0.0, 0.5);
}


       | label '#' label ':' number ':' number ':' number // X#H1:branch_length:support:probability
{
  $$ = reticulation_node_second($3, atof($5), atof($7), atof($9));
  free($5);
  free($7);
  free($9);
}
       | label '#' label ':' number ':' number // X#H1:branch_length:support
{
  $$ = reticulation_node_second($3, atof($5), atof($7), 0.5);
  free($5);
  free($7);
}
       | label '#' label ':' number ':' ':' number // X#H1:branch_length::probability
{
  $$ = reticulation_node_second($3, atof($5), 0.0, atof($8));
  free($5);
  free($8);
}
       | label '#' label ':' ':' number ':' number // X#H1::support:probability
{
  $$ = reticulation_node_second($3, 0.0, atof($6), atof($8));
  free($6);
  free($8);
}
       | label '#' label ':' number // X#H1:branch_length
{
  $$ = reticulation_node_second($3, atof($5), 0.0, 0.5);
  free($5);
}
       | label '#' label ':' ':' number // X#H1::support
{
  $$ = reticulation_node_second($3, 0.0, atof($6), 0.5);
  free($6);
}
       | label '#' label ':' ':' ':' number // X#H1:::probability
{
  $$ = reticulation_node_second($3, 0.0, 0.0, atof($7));
  free($7);
}
       | label '#' label // X#H1
{
  $$ = reticulation_node_second($3, 0.0, 0.0, 0.5);
}


       | '#' label ':' number ':' number ':' number // #H1:branch_length:support:probability
{
  $$ = reticulation_node_second($2, atof($4), atof($6), atof($8));
  free($4);
  free($6);
  free($8);
}
       | '#' label ':' number ':' number // #H1:branch_length:support
{
  $$ = reticulation_node_second($2, atof($4), atof($6), 0.5);
  free($4);
  free($6);
}
       | '#' label ':' number ':' ':' number // #H1:branch_length::probability
{
  $$ = reticulation_node_second($2, atof($4), 0.0, atof($7));
  free($4);
  free($7);
}
       | '#' label ':' ':' number ':' number // #H1::support:probability
{
  $$ = reticulation_node_second($2, 0.0, atof($5), atof($7));
  free($5);
  free($7);
}
       | '#' label ':' number // #H1:branch_length
{
  $$ = reticulation_node_second($2, atof($4), 0.0, 0.5);
  free($4);
}
       | '#' label ':' ':' number // #H1::support
{
  $$ = reticulation_node_second($2, 0.0, atof($5), 0.5);
  free($5);
}
       | '#' label ':' ':' ':' number // #H1:::probability
{
  $$ = reticulation_node_second($2, 0.0, 0.0, atof($6));
  free($6);
}
       | '#' label // #H1
{
  $$ = reticulation_node_second($2, 0.0, 0.0, 0.5);
}

       | label ':' number
{
  $$ = tip_node($1, atof($3));
  free($3);
}

       | label
{
  $$ = tip_node($1, 0.0);
};

%%

rnetwork_t * rnetwork_parse_newick(const char * filename)
{
  rnetwork_t * network;

  /* reset counters */
  tip_cnt = 0;
  inner_tree_cnt = 0;
  reticulation_cnt = 0;
  reticulation_node_pointers = (rnetwork_node_t **)calloc(64, sizeof(rnetwork_node_t*));

  /* open input file */
  rnetwork_in = fopen(filename, "r");
  if (!rnetwork_in)
  {
    printf("Unable to open file (%s)\n", filename);
    
    /* free the counters */
    free(reticulation_node_pointers);
    
    return 0;
  }

  rnetwork_node_t * root = create_rnetwork_node_t();
  /* create root node */
  if (!root)
  {
    printf("Unable to allocate enough memory.\n");
    
    /* free the counters */
    free(reticulation_node_pointers);
    
    return 0;
  }

  if (rnetwork_parse(root))
  {
    rnetwork_graph_destroy(root,NULL);
    root = NULL;
    fclose(rnetwork_in);
    rnetwork_lex_destroy();
    
    /* free the counters */
    free(reticulation_node_pointers);
    
    return 0;
  }

  if (rnetwork_in) fclose(rnetwork_in);

  rnetwork_lex_destroy();

  //initialize clv and scaler indices
  //rnetwork_reset_template_indices(root, tip_cnt);

  /* wrap network */
  network = rnetwork_wrapnetwork(root);
  
  /* free the counters */
  free(reticulation_node_pointers);

  return network;
}

rnetwork_t * rnetwork_parse_newick_string(const char * s)
{
  int rc;
  rnetwork_t * network = NULL;

  /* reset counters */
  tip_cnt = 0;
  inner_tree_cnt = 0;
  reticulation_cnt = 0;
  reticulation_node_pointers = (rnetwork_node_t **)calloc(64, sizeof(rnetwork_node_t*));

  rnetwork_node_t * root = create_rnetwork_node_t();
  if (!root)
  {
    printf("Unable to allocate enough memory.\n");
    
    /* free the counters */
    free(reticulation_node_pointers);
    
    return 0;
  }

  struct rnetwork_buffer_state * buffer = rnetwork__scan_string(s);
  rc = rnetwork_parse(root);
  rnetwork__delete_buffer(buffer);

  rnetwork_lex_destroy();

  if (!rc)
  {
    //initialize clv and scaler indices */
    //rnetwork_reset_template_indices(root, tip_cnt);
    network = rnetwork_wrapnetwork(root);
  }
  else
  {
    free(root);
  }  
  /* free the counters */
  free(reticulation_node_pointers);

  return network;
}
