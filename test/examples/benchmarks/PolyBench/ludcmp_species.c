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
// Filename...........benchmark/ludcmp.c
// Author.............Cedric Nugteren
// Last modified on...23-May-2012
//

#include "common.h"

// This is 'ludcmp', an LU decomposition kernel
int main(void) {
	int i,j,k;
	float w[1];
	
	// Declare arrays on the stack
	float A[N+1][N+1];
	float b[N+1];
	float x[N+1];
	float y[N+1];
	
	// Set the input data
	for (i=0; i<=N; i++) {
		x[i] = i+1;
		y[i] = (i+1)/(float)(N*40) + 1;
		b[i] = (i+1)/(float)(N*20) + 42;
		for (j=0; j<=N; j++) {
			A[i][j] = (i+1)/(float)(10*N) + (j+1)/(float)(5*N);
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		b[0] = 1.0;
		for (i = 0; i < N; i++) {
			for (j = i + 1; j <= N; j++) {
				w[0] = A[j][i];
				for (k = 0; k < i; k++) {
					w[0] = w[0] - A[j][k] * A[k][i];
				}
				A[j][i] = w[0] / A[i][i];
			}
			for (j = i + 1; j <= N; j++) {
				w[0] = A[i + 1][j];
				for (k = 0; k <= i; k++) {
					w[0] = w[0] - A[i + 1][k] * A[k][j];
				}
				A[i + 1][j] = w[0];
			}
		}
		y[0] = b[0];
		for (i = 1; i <= N; i++) {
			w[0] = b[i];
			#pragma species kernel w[0:0]|full ^ A[i:i,0:i-1]|element ^ y[0:i-1]|element -> w[0:0]|shared
			for (j = 0; j < i; j++) {
				w[0] = w[0] - A[i][j] * y[j];
			}
			#pragma species endkernel ludcmp_k18
			y[i] = w[0];
		}
		x[N] = y[N] / A[N][N];
		for (i = 0; i <= N - 1; i++) {
			w[0] = y[N - 1 - i];
			for (j = N - i; j <= N; j++) {
				w[0] = w[0] - A[N - 1 - i][j] * x[j];
			}
			x[N - 1 - i] = w[0] / A[N - 1 - i][N - 1 - i];
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

