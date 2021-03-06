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
// Filename...........benchmark/jacobi-1d-imper.c
// Author.............Cedric Nugteren
// Last modified on...03-April-2012
//

#include "common.h"

// This is 'jacobi-1d-imper', a 1D Jacobi stencil computation
int main(void) {
	int i,j,t;
	
	// Declare arrays on the stack
	float A[LARGE_N];
	float B[LARGE_N];
	
	// Set the input data
	for (i=0; i<LARGE_N; i++) {
		A[i] = ((float) i+2) / LARGE_N;
		B[i] = ((float) i+3) / LARGE_N;
	}
	
	// Perform the computation
	#pragma scop
	{
		for (t = 0; t < TSTEPS; t++) {
			#pragma species kernel A[0:LARGE_N-1]|neighbourhood(-1:1) -> B[1:LARGE_N-2]|element
			for (i = 1; i < LARGE_N - 1; i++) {
				B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
			}
			#pragma species endkernel jacobi-1d-imper_k2
			#pragma species kernel B[1:LARGE_N-2]|element -> A[1:LARGE_N-2]|element
			for (j = 1; j < LARGE_N - 1; j++) {
				A[j] = B[j];
			}
			#pragma species endkernel jacobi-1d-imper_k3
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

