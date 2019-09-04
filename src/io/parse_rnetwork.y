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
%type<s> label optional_label
%type<d> number optional_length optional_number
%type<network> subnetwork
%start input
%%

input: '(' subnetwork ',' subnetwork ')' optional_label optional_length ';'
{
  inner_tree_cnt++;
  network->is_reticulation = 0;
  network->reticulation_index = -1;
  network->left   = $2;
  network->right  = $4;
  network->label  = $6;
  network->length = $7 ? atof($7) : 0;
  network->parent = NULL;
  network->idx = 0;
  free($7);

  if (network->left->is_reticulation)
  {
    if (!network->left->first_parent)
    {
      network->left->first_parent = network;
    }
    else
    {
      network->left->second_parent = network;
    }
  }
  else
  {
    network->left->parent = network;
  }
  
  if (network->right->is_reticulation)
  {
    if (!network->right->first_parent)
    {
      network->right->first_parent = network;
    }
    else
    {
      network->right->second_parent = network;
    }
  }
  else
  {
    network->right->parent = network;
  }
}
       | '(' subnetwork ',' subnetwork ',' subnetwork ')' optional_label optional_length ';'
{
  // unrooted tree/ network -> root it!
  inner_tree_cnt++;
  network->is_reticulation = 0;
  network->reticulation_index = -1;
  network->parent = NULL;
  network->label = $8;
  network->length = $9 ? atof($9) : 0;
  network->idx = 0;
  free($9);
  
  rnetwork_node_t * intermediary_node = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t));
  inner_tree_cnt++;
  intermediary_node->is_reticulation = 0;
  intermediary_node->reticulation_index = -1;
  intermediary_node->label = NULL;
  intermediary_node->length = 0;
  intermediary_node->idx = 0;
  
  intermediary_node->left = $2;
  intermediary_node->right = $4;
  if (intermediary_node->left->is_reticulation)
  {
    if (!intermediary_node->left->first_parent)
    {
      intermediary_node->left->first_parent = intermediary_node;
    }
    else
    {
      intermediary_node->left->second_parent = intermediary_node;
    }
  }
  else
  {
    intermediary_node->left->parent = intermediary_node;
  }
  
  if (intermediary_node->right->is_reticulation)
  {
    if (!intermediary_node->right->first_parent)
    {
      intermediary_node->right->first_parent = intermediary_node;
    }
    else
    {
      intermediary_node->right->second_parent = intermediary_node;
    }
  }
  else
  {
    intermediary_node->right->parent = intermediary_node;
  }
  
  network->left = intermediary_node;
  network->right = $6;
  if (network->right->is_reticulation)
  {
    if (!network->right->first_parent)
    {
      network->right->first_parent = network;
    }
    else
    {
      network->right->second_parent = network;
    }
  }
}
;

subnetwork: '(' subnetwork ',' subnetwork ')' optional_label optional_length
{
  inner_tree_cnt++;
  $$ = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t));
  $$->is_reticulation = 0;
  $$->reticulation_index = -1;
  $$->left   = $2;
  $$->right  = $4;
  $$->label  = $6;
  $$->length = $7 ? atof($7) : 0;
  free($7);
  $$->idx = 0;

  if ($$->left->is_reticulation)
  {
    if (!$$->left->first_parent)
    {
      $$->left->first_parent = $$;
    }
    else
    {
      $$->left->second_parent = $$;
    }
  }
  else
  {
    $$->left->parent = $$;
  }
  
  if ($$->right->is_reticulation)
  {
    if (!$$->right->first_parent)
    {
      $$->right->first_parent = $$;
    }
    else
    {
      $$->right->second_parent = $$;
    }
  }
  else
  {
    $$->right->parent = $$;
  }
}
       | '(' subnetwork ')' optional_label '#' label ':' optional_number ':' optional_number ':' number // optional_branch_length, optional_support, prob
{
  $$ = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t));
  $$->is_reticulation = 1;
  $$->child   = $2;
  $$->left   = NULL;
  $$->right  = NULL;
  $$->first_parent   = NULL;
  $$->second_parent  = NULL;
  $$->label  = $4;
  $$->reticulation_name = $6;
  $$->reticulation_index = reticulation_cnt;
  if ($8)
  {
    $$->first_parent_length = atof($8);
    free($8);
  }
  else
  {
    $$->first_parent_length = 0;
  }
  if ($10)
  {
    $$->support = atof($10);
    free($10);
  }
  else
  {
    $$->support = 0;
  }
  if ($12)
  {
    $$->first_parent_prob = atof($12);
    free($12);
  }
  else
  {
    $$->first_parent_prob = 0.5;
  }
  $$->idx = 0;

  $$->child->parent  = $$;

  reticulation_node_pointers[reticulation_cnt] = $$;
  reticulation_node_names[reticulation_cnt] = $$->reticulation_name;
  reticulation_cnt++;
}

       | '(' subnetwork ')' optional_label '#' label ':' optional_number ':' number // optional_branch_length, support
{
  $$ = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t));
  $$->is_reticulation = 1;
  $$->child   = $2;
  $$->left   = NULL;
  $$->right  = NULL;
  $$->first_parent   = NULL;
  $$->second_parent  = NULL;
  $$->label  = $4;
  $$->reticulation_name = $6;
  $$->reticulation_index = reticulation_cnt;
  if ($8)
  {
    $$->first_parent_length = atof($8);
    free($8);
  }
  else
  {
    $$->first_parent_length = 0;
  }

  $$->support = atof($10);
  free($10);

  $$->first_parent_prob = 0.5;

  $$->idx = 0;

  $$->child->parent  = $$;

  reticulation_node_pointers[reticulation_cnt] = $$;
  reticulation_node_names[reticulation_cnt] = $$->reticulation_name;
  reticulation_cnt++;
}

       | '(' subnetwork ')' optional_label '#' label // no branch length, no support, no prob
{
  $$ = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t));
  $$->is_reticulation = 1;
  $$->child   = $2;
  $$->left   = NULL;
  $$->right  = NULL;
  $$->first_parent   = NULL;
  $$->second_parent  = NULL;
  $$->label  = $4;
  $$->reticulation_name = $6;
  $$->reticulation_index = reticulation_cnt;
  $$->length = 0;
  $$->support = 0;
  $$->idx = 0;

  $$->child->parent  = $$;
  reticulation_node_pointers[reticulation_cnt] = $$;
  reticulation_node_names[reticulation_cnt] = $$->reticulation_name;
  reticulation_cnt++;
}
       | '(' subnetwork ')' optional_label '#' label ':' number // only branch length
{
  $$ = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t));
  $$->is_reticulation = 1;
  $$->child   = $2;
  $$->left   = NULL;
  $$->right  = NULL;
  $$->first_parent   = NULL;
  $$->second_parent  = NULL;
  $$->label  = $4;
  $$->reticulation_name = $6;
  $$->reticulation_index = reticulation_cnt;
  if ($8) {
    $$->length = atof($8);
    free($8);
  } else {
    $$->length = 0;
  }
  $$->support = 0;
  $$->idx = 0;

  $$->child->parent  = $$;
  reticulation_node_pointers[reticulation_cnt] = $$;
  reticulation_node_names[reticulation_cnt] = $$->reticulation_name;
  reticulation_cnt++;
}
       | optional_label '#' label ':' optional_number ':' optional_number ':' number // branch length, support, probability
{
  unsigned int i = 0;
  for (i = 0; i < reticulation_cnt; ++i)
  {
    if (strcmp(reticulation_node_names[i],$3) == 0)
    {
      $$ = reticulation_node_pointers[i];
      if ($5) {
        $$->second_parent_length = atof($5);
        free($5);
      } else {
        $$->second_parent_length = 0;
      }
      if ($7) {
        $$->support = atof($7);
        //free($7); // TODO: why does this line cause double free or corruption?
      } else {
        $$->support = 0;
      }
      if ($9) {
        $$->second_parent_prob = atof($9);
        free($9); 
      } else {
        $$->second_parent_prob = 1.0 - $$->first_parent_prob;
      }
      break;
    }
  }
}

       | optional_label '#' label ':' optional_number ':' number // optional_branch length, support
{
  unsigned int i = 0;
  for (i = 0; i < reticulation_cnt; ++i)
  {
    if (strcmp(reticulation_node_names[i],$3) == 0)
    {
      $$ = reticulation_node_pointers[i];
      if ($5) {
        $$->second_parent_length = atof($5);
        free($5);
      } else {
        $$->second_parent_length = 0;
      }
      $$->support = atof($7);
      free($7);
      
      $$->second_parent_prob = 1.0 - $$->first_parent_prob;
      break;
    }
  }
}

       | optional_label '#' label optional_length
{
  unsigned int i = 0;
  for (i = 0; i < reticulation_cnt; ++i)
  {
    if (strcmp(reticulation_node_names[i],$3) == 0)
    {
      $$ = reticulation_node_pointers[i];
      if ($4) {
        $$->second_parent_length = atof($4);
        free($4);
      } else {
        $$->second_parent_length = 0;
      }
      $$->support = 0;
      $$->second_parent_prob = 1.0 - $$->first_parent_prob;
      break;
    }
  }
}
       | label optional_length
{
  $$ = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t));
  $$->is_reticulation = 0;
  $$->reticulation_index = -1;
  $$->label  = $1;
  $$->length = $2 ? atof($2) : 0;
  $$->left   = NULL;
  $$->right  = NULL;
  $$->first_parent   = NULL;
  $$->second_parent  = NULL;
  $$->child = NULL;
  $$->idx = 0;
  tip_cnt++;
  free($2);
};


optional_label:  {$$ = NULL;} | label  {$$ = $1;};
optional_length: {$$ = NULL;} | ':' number {$$ = $2;};
label: STRING    {$$=$1;} | NUMBER {$$=$1;};
number: NUMBER   {$$=$1;};

optional_number: {$$ = NULL;} | number {$$ = $1;};

%%

rnetwork_t * rnetwork_parse_newick(const char * filename)
{
  rnetwork_t * network;

  struct rnetwork_node_s * root;

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

  /* create root node */
  if (!(root = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t))))
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
  struct rnetwork_node_s * root;
  rnetwork_t * network = NULL;

  /* reset counters */
  tip_cnt = 0;
  inner_tree_cnt = 0;
  reticulation_cnt = 0;
  reticulation_node_pointers = (rnetwork_node_t **)calloc(64, sizeof(rnetwork_node_t*));

  if (!(root = (rnetwork_node_t *)calloc(1, sizeof(rnetwork_node_t))))
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
