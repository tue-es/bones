//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example application: 'Speckle Reducing
// Anisotropic Diffusion' or 'SRAD', taken from the Rodinia benchmark suite. For
// more information on the application or on Bones please use the contact infor-
// mation below.
//
// == More information on SRAD (Speckle Reducing Anisotropic Diffusion):
// Article............http://dx.doi.org/10.1109/TIP.2002.804276
// Original code......https://www.cs.virginia.edu/~skadron/wiki/rodinia/
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/srad.c
// Authors............Cedric Nugteren
// Original authors...Rob Janiczek, Drew Gilliam, Lukasz Szafaryn
// Last modified on...10-Aug-2012
//

//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//########################################################################
//### Defines
//########################################################################

#define ROWS 128    // Number of ROWS in the domain
#define COLS 128    // Number of COLS in the domain
#define R1 0        // y1 position of the speckle
#define R2 31       // y2 position of the speckle
#define C1 0        // x1 position of the speckle
#define C2 31       // x2 position of the speckle
#define LAMBDA 0.5  // Lambda value
#define NITER 2     // Number of iterations

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {
	
	// Declare the loop iterators
	int i,j,iter;
	
	// Declare domain variables
	float mean_roi, var_roi;
	float q0s, qs;
	float divergence;
	float cN, cS, cW, cE;
	float G2, L;
	
	// Declare other/helper variables
	int index;
	float temp_value;
	float sum1, sum2;
	float current_value;
	float temp_a, temp_b;
	
	// Check for valid row and column sizes
	if ((ROWS%16 != 0 ) || (COLS%16 != 0)) {
		printf("[srad] Error: the number of rows and columns must be multiples of 16\n");
		fflush(stdout); exit(1);
	}
	
	// Initialising memory
	printf("\n[srad] Initialising memory"); fflush(stdout);
	int size = COLS*ROWS;
	int size_roi = (R2-R1+1)*(C2-C1+1);
	float* values     = (float*) malloc(sizeof(float)*size);
	float* coefficent = (float*) malloc(sizeof(float)*size);
	float* dN = (float*) malloc(sizeof(float)*size);
	float* dS = (float*) malloc(sizeof(float)*size);
	float* dW = (float*) malloc(sizeof(float)*size);
	float* dE = (float*) malloc(sizeof(float)*size);
	
	// Populate the input matrix
	printf("\n[srad] Populating the input matrix with random values"); fflush(stdout);
	for (i=0; i<ROWS; i++) {
		for (j=0; j<COLS; j++) {
			temp_value = rand()/(float)RAND_MAX;
			values[i*COLS+j] = (float)exp(temp_value);
		}
	}
	
	// Perform the computation a given number of times
	printf("\n[srad] Performing the computation %d times",NITER); fflush(stdout);
	for (iter=0; iter<NITER; iter++) {
		
		// Compute the mean, the variance and the speckle scale function (q0s) of the region of interest (ROI)
		sum1 = 0;
		sum2 = 0;
		for (i=R1; i<=R2; i++) {
			for (j=C1; j<=C2; j++) {
				temp_value = values[i*COLS+j];
				sum1 += temp_value;
				sum2 += temp_value*temp_value;
			}
		}
		mean_roi = sum1/size_roi;
		var_roi = (sum2/size_roi) - mean_roi*mean_roi;
		q0s = var_roi / (mean_roi*mean_roi);
		
		// Iterate over the full image and compute
		#pragma scop
		for (i=0; i<ROWS; i++) {
			for (j=0; j<COLS; j++) {
				index = i*COLS+j;
				current_value = values[index];

				// Compute the directional derivates (N,S,W,E)
				if (i==0)      { dN[index] = 0; }
				else           { dN[index] = values[(i-1)*COLS + j    ] - current_value; }
				if (i==ROWS-1) { dS[index] = 0; }
				else           { dS[index] = values[(i+1)*COLS + j    ] - current_value; }
				if (j==0)      { dW[index] = 0; }
				else           { dW[index] = values[i    *COLS + (j-1)] - current_value; }
				if (j==COLS-1) { dE[index] = 0; }
				else           { dE[index] = values[i    *COLS + (j+1)] - current_value; }

				// Compute the instantaneous coefficient of variation (qs) (equation 35)
				G2 = (dN[index]*dN[index] + dS[index]*dS[index] + dW[index]*dW[index] + dE[index]*dE[index]) / (current_value*current_value);
				L =  (dN[index]           + dS[index]           + dW[index]           + dE[index]          ) / (current_value              );
				temp_a = (0.5*G2)-((1.0/16.0)*(L*L));
				temp_b = 1+(0.25*L);
				qs = temp_a/(temp_b*temp_b);

				// Set the diffusion coefficent (equation 33)
				coefficent[index] = 1.0 / (1.0+( (qs-q0s)/(q0s*(1+q0s)) ));

				// Saturate the diffusion coefficent
				if (coefficent[index] < 0) {
					coefficent[index] = 0;
				}
				else if (coefficent[index] > 1) {
					coefficent[index] = 1;
				}
			}
		}
		
		// Iterate over the full image again and compute the final values
		for (i=0; i<ROWS; i++) {
			for (j=0; j<COLS; j++) {
				index = i*COLS+j;

				// Calculate the diffusion coefficent
				                 cN = coefficent[i    *COLS+j    ];
				if (i==ROWS-1) { cS = 0; }
				else           { cS = coefficent[(i+1)*COLS+j    ]; }
				                 cW = coefficent[i    *COLS+j    ];
				if (j==COLS-1) { cE = 0; }
				else           { cE = coefficent[i    *COLS+(j+1)]; }

				// Calculate the divergence (equation 58)
				divergence = cN*dN[index] + cS*dS[index] + cW*dW[index] + cE*dE[index];

				// Update the image accordingly (equation 61)
				values[index] = values[index] + 0.25*LAMBDA*divergence;
			}
		}
		#pragma endscop
	}
	
	// Print the values matrix
	printf("\n[srad] Printing the output matrix:\n\n"); fflush(stdout);
	for (i=0; i<ROWS; i++) {
		for (j=0; j<COLS; j++) {
			printf("%.5f ", values[i*COLS+j]);
		}
		printf("\n");
	}
	
	// Clean-up and exit
	printf("\n[srad] Completed\n\n"); fflush(stdout);
	free(values); free(coefficent);
	free(dN); free(dS); free(dW); free(dE);
	fflush(stdout);
	return 0;
}

//########################################################################