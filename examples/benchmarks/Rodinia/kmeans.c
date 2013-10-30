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
// Last modified on...10-Aug-2012
//

//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <math.h>
#include <float.h>

//########################################################################
//### Defines
//########################################################################

#define SIZE 512
#define NUM_CLUSTERS 20
#define DIMENSIONS 2
#define THRESHOLD 0.0001

//########################################################################
//### Start of the main function
//########################################################################
int main(void) {
	
	// Declare the loop iterators
	int i,j,k;
	
	// Declare the error variables
	double error = DBL_MAX;
	double old_error;
	int iterations = 0;
	
	// Declare the distance variables and arrays
	double distance[1];
	double min_distance[1];
	double distances[SIZE];
	
	// Initialising memory
	printf("\n[k-means] Initialising memory"); fflush(stdout);
	double input[SIZE][DIMENSIONS];
	double centroids[NUM_CLUSTERS][DIMENSIONS];
	double centroids_temp[NUM_CLUSTERS][DIMENSIONS];
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
		#pragma scop
		
		// Save the error from the last step
		old_error = error;
		error = 0;
		
		// Clear old counts and temporary centroids
		for (k=0; k<NUM_CLUSTERS; k++) {
			counts[k] = 0;
			for (j=0; j<DIMENSIONS; j++) {
				centroids_temp[k][j] = 0;
			}
		}
		
		// Iterate over all data points
		for (i=0; i<SIZE; i++) {
			
			// Find the closest cluster
			min_distance[0] = DBL_MAX;
			for (k=0; k<NUM_CLUSTERS; k++) {
				distance[0] = 0;
				for (j=0; j<DIMENSIONS; j++) {
					distance[0] += pow(input[i][j]-centroids[k][j],2);
				}
				if (distance[0] < min_distance[0]) {
					output[i] = k;
					min_distance[0] = distance[0];
				}
			}
			
			// Update the size and temporary centroid of the destination cluster
			for (j=0; j<DIMENSIONS; j++) {
				centroids_temp[output[i]][j] += input[i][j];
			}
			counts[output[i]] += 1;
			
			// Store the resulting distance
			distances[i] = min_distance[0];
		}
		
		// Update the standard error
		for (i=0; i<SIZE; i++) {
			error += distances[i];
		}
		
		// Update all centroids
		for (k=0; k<NUM_CLUSTERS; k++) {
			for (j=0; j<DIMENSIONS; j++) {
				if (counts[k]) {
					centroids[k][j] = centroids_temp[k][j] / counts[k];
				}
				else {
					centroids[k][j] = centroids_temp[k][j];
				}
			}
		}
		
		// Go to the next iteration
		iterations += 1;
		
		#pragma endscop
	} while (fabs(error-old_error) > THRESHOLD);
	
	// Print the results
	printf("\n[k-means] Algorithm finished in %d iterations with an error of %.3lf", iterations, error); fflush(stdout);
	printf("\n[k-means] Printing the results: \n\n"); fflush(stdout);
	for (k=0; k<NUM_CLUSTERS; k++) {
		printf("Cluster %2i: ", k);
		for (i=0; i<SIZE; i++) {
			if (output[i] == k) {
				printf("%3i ", i);
			}
		}
		printf("\n");
	}
	
	// Clean-up and exit the function
	printf("\n[k-means] Completed\n\n"); fflush(stdout);
	fflush(stdout);
	return 0;
}

//########################################################################