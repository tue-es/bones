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
// Filename...........benchmark/trisolv.c
// Author.............Cedric Nugteren
// Last modified on...03-Jul-2012
//

#include "common.h"

// This is 'trisolv', a solver for linear systems with triangular matrices
int main(void) {
	int i,j;
	float A_i_i;
	
	// Declare arrays on the stack
	float A[NX][NX];
	float c[NX];
	float x[NX];
	
	// Set the input data
	for (i=0; i<NX; i++) {
		c[i] = ((float) i) / NX;
		x[i] = ((float) i) / NX;
		for (j=0; j<NX; j++) {
			A[i][j] = ((float) i*j) / NX;
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		for (i = 0; i < NX; i++) {
			x[i] = c[i];
			A_i_i = A[i][i];
			#pragma species kernel x[i:i]|full ^ A[i:i,0:i-1]|element ^ x[0:i-1]|element -> x[i:i]|shared
			for (j = 0; j <= i - 1; j++) {
				x[i] = x[i] - A[i][j] * x[j];
				x[i] = x[i] / A_i_i;
			}
			#pragma species endkernel trisolv_k4
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}
