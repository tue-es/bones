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
// Filename...........benchmark/trmm.c
// Author.............Cedric Nugteren
// Last modified on...26-Jun-2012
//

#include "common.h"

// This is 'trmm', a triangular matrix multiplication kernel
int main(void) {
	int i,j,k;
	
	// Declare arrays on the stack
	float A[NI][NI];
	float B[NI][NI];
	
	// Set the constants
	int alpha = 32412;
	
	// Set the input data
	for (i=0; i<NI; i++) {
		for (j=0; j<NI; j++) {
			A[i][j] = ((float) i*j) / NI;
			B[i][j] = ((float) i*j) / NI;
		}
	}
	
	// Perform the computation (B := alpha*A'*B, with A triangular)
	#pragma scop
	{
		for (i = 1; i < NI; i++) {
			for (j = 0; j < NI; j++) {
				#pragma species kernel B[i:i,j:j]|full ^ A[i:i,0:i-1]|element ^ B[j:j,0:i-1]|element -> B[i:i,j:j]|shared
				for (k = 0; k < i; k++) {
					B[i][j] += alpha * A[i][k] * B[j][k];
				}
				#pragma species endkernel trmm_k3
			}
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

