//
// This file is part of the Bones source-to-source compiler examples. The C-code
// is largely identical in terms of functionality and variable naming to the code
// found in PolyBench/C version 3.2. For more information on PolyBench/C or Bones
// please use the contact information below.
//
// == More information on PolyBench/C
// Contact............Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address........http://polybench.sourceforge.net/
// 
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........benchmark/covariance.c
// Author.............Cedric Nugteren
// Last modified on...26-Jun-2012
//

#include "common.h"

// This is 'covariance', a covariance computation algorithm
int main(void) {
	int i,j,j1,j2;
	
	// Declare arrays on the stack
	float data[N][M];
	float mean[M];
	float symmat[M][M];
	
	// Set the constants
	float float_n = 1.2;
	
	// Set the input data
/*	for (i=0; i<N; i++) {
		for (j=0; j<M; j++) {
			data[i][j] = ((float) i*j) / M;
		}
	}
*/	
	// Perform the computation
	#pragma scop
	#pragma species kernel 0:N-1,0:M-1|chunk(0:N-1,0:0) -> 0:M-1|element
	// Determine the mean of the column vectors of the input data matrix
	for (j=0; j<M; j++) {
		mean[j] = 0.0;
		for (i=0; i<N; i++) {
			mean[j] += data[i][j];
		}
		mean[j] /= float_n;
	}
	#pragma species endkernel covariance-part1
	#pragma species kernel 0:N-1,0:M-1|element ^ 0:M-1|element -> 0:N-1,0:M-1|element
	// Center the column vectors
	for (i=0; i<N; i++) {
		for (j=0; j<M; j++) {
			data[i][j] -= mean[j];
		}
	}
	#pragma species endkernel covariance-part2
	
	#pragma endscop
	
	// Calculate the MxM covariance matrix
	for (j1=0; j1<M; j1++) {
		#pragma species kernel 0:N-1,j1:j1|full ^ 0:N-1,j1:M-1|chunk(0:N-1,0:0) -> j1:M-1,j1:j1|element ^ j1:j1,j1:M-1|element
		for (j2=j1; j2<M; j2++) {
			symmat[j1][j2] = 0.0;
			for (i=0; i<N; i++) {
				symmat[j1][j2] += data[i][j1] * data[i][j2];
			}
			symmat[j2][j1] = symmat[j1][j2];
		}
		#pragma species endkernel covariance-part3
	}
	
	// Clean-up and exit the function
	fflush(stdout);
	symmat[8][9] = symmat[8][9];
	return 0;
}
