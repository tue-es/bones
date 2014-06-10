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
// Filename...........benchmark/cholesky.c
// Author.............Cedric Nugteren
// Last modified on...03-Jul-2012
//

#include "common.h"

// This is 'cholesky', a cholesky decomposition kernel
int main(void) {
	int i,j,k;
	float x[1];
	float p_i;
	
	// Declare arrays on the stack
	float A[N][N];
	float p[N];
	
	// Set the input data
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			A[i][j] = i*2.3 + 1;
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		for (i = 0; i < N; i++) {
			x[0] = A[i][i];
			#pragma species kernel x[0:0]|full ^ A[i:i,0:i-1]|element -> x[0:0]|shared
			for (j = 0; j <= i - 1; j++) {
				x[0] = x[0] - A[i][j] * A[i][j];
			}
			#pragma species endkernel cholesky_k3
			p[i] = 1.0 / sqrt(x[0]);
			p_i = p[i];
			#pragma species kernel A[i:i,N-1:i+1]|element ^ A[N-1:i+1,0:i-1]|chunk(0:0,0:i-1) ^ A[i:i,0:i-1]|full -> x[0:0]|shared ^ A[N-1:i+1,i:i]|element
			for (j = i + 1; j < N; j++) {
				x[0] = A[i][j];
				for (k = 0; k <= i - 1; k++) {
					x[0] = x[0] - A[j][k] * A[i][k];
				}
				A[j][i] = x[0] * p_i;
			}
			#pragma species endkernel cholesky_k6
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}
