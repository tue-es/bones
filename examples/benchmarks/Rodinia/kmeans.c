//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example application: 'K-means clustering',
// as also available in the Rodinia benchmark suite. For more information on the
// application or on Bones please use the contact information below.
//
// == This implementation of K-means clustering is inspired by:
// Author.............Roger Zhang
// Web address........http://cs.smu.ca/~r_zhang/code/kmeans.c
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/kmeans.c
// Authors............Cedric Nugteren
// Last modified on...06-Jun-2014
//
//########################################################################

// Includes
#include "common.h"

//########################################################################
//### Start of the main function
//########################################################################
int main(void) {
	
	// Declare the loop iterators
	int i,j,k;
	
	// Declare the error variables
	float error[1] = { 100000.0 };
	float old_error;
	int iterations = 0;
	
	// Declare the distance variables and arrays
	float distances[SIZE];
	
	// Initialising memory
	printf("\n[k-means] Initialising memory"); fflush(stdout);
	float input[SIZE][DIMENSIONS];
	float centroids[NUM_CLUSTERS][DIMENSIONS];
	float centroids_temp[NUM_CLUSTERS][DIMENSIONS];
	int output[SIZE];
	int counts[NUM_CLUSTERS];
	
	// Set the input data
	printf("\n[k-means] Populating memory"); fflush(stdout);
	for (i=0; i<SIZE; i++) {
		input[i][0] = (i/16);
		input[i][1] = i%4;
	}
	
	// Pick k initial centroids
	printf("\n[k-means] Setting 'k' initial centroids"); fflush(stdout);
	for (k=0; k<NUM_CLUSTERS; k++) {
		for (j=0; j<DIMENSIONS; j++) {
			centroids[k][j] = input[(SIZE/NUM_CLUSTERS)*k][j];
		}
	}
	
	// Perform the k-means clustering algorithm, end when the error is not becoming smaller
	printf("\n[k-means] Perform the clustering algorithm"); fflush(stdout);
	do {
	//for (int iters=0; iters<10; iters++) {
		
		// Start of the scop
		#pragma scop
		
		// Save the error from the last step
		old_error = error[0];
		error[0] = 0;

		// Clear old counts and temporary centroids
		for (k=0; k<NUM_CLUSTERS; k++) {
			counts[k] = 0;
		}
		for (k=0; k<NUM_CLUSTERS; k++) {
			for (j=0; j<DIMENSIONS; j++) {
				centroids_temp[k][j] = 0;
			}
		}
		// Iterate over all data points
		for (i=0; i<SIZE; i++) {
			
			// Find the closest cluster
			float min_distance = 100000.0;
			for (k=0; k<NUM_CLUSTERS; k++) {
				float distance = 0;
				for (j=0; j<DIMENSIONS; j++) {
					float val = (input[i][j]-centroids[k][j]);
					distance += val * val;
				}
				if (distance < min_distance) {
					output[i] = k;
					min_distance = distance;
				}
			}
			
			// Update the size and temporary centroid of the destination cluster
			int cluster_index = output[i];
			for (j=0; j<DIMENSIONS; j++) {
				centroids_temp[cluster_index][j] += input[i][j];
			}
			counts[cluster_index] += 1;
			
			// Store the resulting distance
			distances[i] = min_distance;
		}
		
		// Update the standard error
		for (i=0; i<SIZE; i++) {
			error[0] += distances[i];
		}
		
		// Update all centroids
		for (k=0; k<NUM_CLUSTERS; k++) {
			for (j=0; j<DIMENSIONS; j++) {
				int count = counts[k];
				float val;
				if (count > 0) {
					val = centroids_temp[k][j] / count;
				}
				else {
					val = centroids_temp[k][j];
				}
				centroids[k][j] = val;
			}
		}
		
		// Go to the next iteration
		iterations += 1;
		#pragma endscop
	//}
	} while (fabs(error[0]-old_error) > THRESHOLD);
	
	// Print the results
	printf("\n[k-means] Algorithm finished in %d iterations with an error of %.3lf", iterations, error[0]); fflush(stdout);
	//printf("\n[k-means] Printing the results: \n\n"); fflush(stdout);
	//for (k=0; k<NUM_CLUSTERS; k++) {
	//	printf("Cluster %2i: ", k);
	//	for (i=0; i<SIZE; i++) {
	//		if (output[i] == k) {
	//			printf("%3i ", i);
	//		}
	//	}
	//	printf("\n");
	//}
	
	// Clean-up and exit the function
	printf("\n[k-means] Completed\n\n"); fflush(stdout);
	fflush(stdout);
	return 0;
}
//########################################################################
