//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example application: 'bfs', taken from
// the Rodinia benchmark suite. For more information on the application or on Bones
// please use the contact information below.
//
// == More information on Hotspot
// Original code......https://www.cs.virginia.edu/~skadron/wiki/rodinia/
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/bfs.c
// Authors............Cedric Nugteren
// Last modified on...08-Jun-2014
//
//########################################################################

// Includes
#include "common.h"

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {
	int no_of_nodes;

	// Read input data
	printf("[bfs] Reading File\n");
	FILE* fp = fopen(FILENAME, "r");
	if (!fp) {
		printf("[bfs] Error Reading graph file\n");
		return 1;
	}
	fscanf(fp,"%d",&no_of_nodes);

	// Arrays
	int h_graph_nodes_start[MAX_NODES];
	int h_graph_nodes_edges[MAX_NODES];
	int h_graph_mask[MAX_NODES];
	int h_updating_graph_mask[MAX_NODES];
	int h_graph_visited[MAX_NODES];

	// Initialize
	int start;
	int edges;
	for (int i=0; i<no_of_nodes; i++) {
		fscanf(fp, "%d %d", &start, &edges);
		h_graph_nodes_start[i] = start;
		h_graph_nodes_edges[i] = edges;
		h_graph_mask[i] = 0;
		h_updating_graph_mask[i] = 0;
		h_graph_visited[i] = 0;
	}

	// Read the source node from the file
	int source = 0;
	fscanf(fp, "%d", &source);
	source = 0;

	// Set the source node as true in the mask
	h_graph_mask[source] = 1;
	h_graph_visited[source] = 1;

	// Get the edge list
	int id;
	int cost;
	int edge_list_size;
	fscanf(fp,"%d",&edge_list_size);
	int h_graph_edges[MAX_NODES];
	for(int i=0; i<edge_list_size; i++) {
		fscanf(fp, "%d", &id);
		fscanf(fp, "%d", &cost);
		h_graph_edges[i] = id;
	}

	// Memory for the result
	int h_cost[MAX_NODES];
	for(int i=0; i<MAX_NODES; i++) {
		h_cost[i] = -1;
	}
	h_cost[source] = 0;
	
	// Start the computation
	printf("[bfs] Start traversing the tree\n");
	int k = 0;
	int stop[1];

	// If no thread changes this value then the loop stops
	stop[0] = 0;

	#pragma scop
	for (unsigned t=0; t<10; t++) {
	//do {

		// Atomic update loop
		for(int tid=0; tid<no_of_nodes; tid++) {
			int val1 = h_graph_mask[tid];
			if (val1 == 1) { 
				h_graph_mask[tid] = 0;
				int val2 = h_graph_nodes_start[tid];
				int val3 = h_graph_nodes_edges[tid];
				for (int i=val2; i<(val3 + val2); i++) {
					int id = h_graph_edges[i];
					int val4 = h_graph_visited[id];
					if (val4 == 0) {
						h_cost[id] = h_cost[tid] + 1;
						h_updating_graph_mask[id] = 1;
					}
				}
			}
		}

		// Atomic update loop
		for (int tid=0; tid<no_of_nodes; tid++) {
			int val1 = h_updating_graph_mask[tid];
			if (val1 == 1) {
				h_graph_mask[tid] = 1;
				h_graph_visited[tid] = 1;
				h_updating_graph_mask[tid] = 0;
				stop[0] = 1;
			}
		}

		// Next iteration
		//k++;
	//} while(stop[0] != 0);
	}
	#pragma endscop

	// Clean-up and exit
	if (fp) {
		fclose(fp);
	}
	printf("\n[bfs] Completed\n\n"); fflush(stdout);
	fflush(stdout);
	return 0;
}

//########################################################################