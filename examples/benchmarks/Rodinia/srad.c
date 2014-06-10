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
// Last modified on...05-Jun-2014
//
//########################################################################

// Includes
#include "common.h"

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
	float values[ROWS][COLS];
	float coefficent[ROWS][COLS];
	float dN[ROWS][COLS];
	float dS[ROWS][COLS];
	float dW[ROWS][COLS];
	float dE[ROWS][COLS];
	
	// Populate the input matrix
	printf("\n[srad] Populating the input matrix with random values"); fflush(stdout);
	for (i=0; i<ROWS; i++) {
		for (j=0; j<COLS; j++) {
			temp_value = rand()/(float)RAND_MAX;
			values[i][j] = (float)exp(temp_value);
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
				temp_value = values[i][j];
				sum1 += temp_value;
				sum2 += temp_value*temp_value;
			}
		}
		mean_roi = sum1/size_roi;
		var_roi = (sum2/size_roi) - mean_roi*mean_roi;
		q0s = var_roi / (mean_roi*mean_roi);
		
		#pragma scop
		// Iterate over the full image and computeΩ
		for (i=0; i<ROWS; i++) {
			for (j=0; j<COLS; j++) {

				current_value = values[i][j];

				// Temporary variables
				float valN = 0;
				float valS = 0;
				float valW = 0;
				float valE = 0;

				// Compute the directional derivates (N,S,W,E)
				if (i > 0)      { valN = values[i-1][j  ] - current_value; }
				if (i < ROWS-1) { valS = values[i+1][j  ] - current_value; }
				if (j > 0)      { valW = values[i  ][j-1] - current_value; }
				if (j < COLS-1) { valE = values[i  ][j+1] - current_value; }

				// Compute the instantaneous coefficient of variation (qs) (equation 35)
				G2 = (valN*valN + valS*valS + valW*valW + valE*valE) / (current_value*current_value);
				L =  (valN      + valS      + valW      + valE     ) / (current_value              );
				temp_a = (0.5*G2)-((1.0/16.0)*(L*L));
				temp_b = 1+(0.25*L);
				qs = temp_a/(temp_b*temp_b);

				// Write the data
				dN[i][j] = valN;
				dS[i][j] = valS;
				dW[i][j] = valW;
				dE[i][j] = valE;

				// Set the diffusion coefficent (equation 33)
				float val = 1.0 / (1.0+( (qs-q0s)/(q0s*(1+q0s)) ));

				// Saturate the diffusion coefficent
				if (val < 0) {
					val = 0;
				}
				else if (val > 1) {
					val = 1;
				}
				coefficent[i][j] = val;
			}
		}

		// Iterate over the full image again and compute the final values
		for (i=0; i<ROWS; i++) {
			for (j=0; j<COLS; j++) {

				// Calculate the diffusion coefficent
				                 cN = coefficent[i  ][j  ];
				if (i==ROWS-1) { cS = 0; }
				else           { cS = coefficent[i+1][j  ]; }
				                 cW = coefficent[i  ][j  ];
				if (j==COLS-1) { cE = 0; }
				else           { cE = coefficent[i  ][j+1]; }

				// Calculate the divergence (equation 58)
				divergence = cN*dN[i][j] + cS*dS[i][j] + cW*dW[i][j] + cE*dE[i][j];

				// Update the image accordingly (equation 61)
				values[i][j] = values[i][j] + 0.25*LAMBDA*divergence;
			}
		}
		#pragma endscop
	}
	
	// Print the values matrix
	printf("\n[srad] Printing the output matrix:\n\n"); fflush(stdout);
	for (i=0; i<ROWS; i++) {
		for (j=0; j<COLS; j++) {
			if (i == 5 && j == 5) {
				printf("%.5f ", values[i][j]);
			}
		}
		//printf("\n");
	}
	
	// Clean-up and exit
	printf("\n[srad] Completed\n\n");
	fflush(stdout);
	return 0;
}

//########################################################################