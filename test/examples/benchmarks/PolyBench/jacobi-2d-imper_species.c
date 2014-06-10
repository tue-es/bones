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
// Filename...........benchmark/jacobi-2d-imper.c
// Author.............Cedric Nugteren
// Last modified on...03-April-2012
//

#include "common.h"

// This is 'jacobi-2d-imper', a 2D Jacobi stencil computation
int main(void) {
	int i,j,t;
	
	// Declare arrays on the stack
	float A[N][N];
	float B[N][N];
	
	// Set the input data
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			A[i][j] = ((float) i*(j+2) + 2) / N;
			B[i][j] = ((float) i*(j+3) + 3) / N;
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		for (t = 0; t < TSTEPS; t++) {
			#pragma species kernel A[0:N-1,0:N-1]|neighbourhood(-1:1,-1:1) -> B[1:N-2,1:N-2]|element
			for (i = 1; i < N - 1; i++) {
				for (j = 1; j < N - 1; j++) {
					B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
				}
			}
			#pragma species endkernel jacobi-2d-imper_k2
			#pragma species kernel B[1:N-2,1:N-2]|element -> A[1:N-2,1:N-2]|element
			for (i = 1; i < N - 1; i++) {
				for (j = 1; j < N - 1; j++) {
					A[i][j] = B[i][j];
				}
			}
			#pragma species endkernel jacobi-2d-imper_k3
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

