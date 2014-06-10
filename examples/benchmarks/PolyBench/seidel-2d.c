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
// Filename...........benchmark/seidel-2d.c
// Author.............Cedric Nugteren
// Last modified on...05-April-2012
//

#include "common.h"

// This is 'seidel-2d', a 2D Seidel stencil computation
int main(void) {
	int i,j,t;
	
	// Declare arrays on the stack
	float A[N][N];
	
	// Set the input data
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			A[i][j] = ((float) i*i*(j+2) + 2) / N;
		}
	}
	
	// Perform the computation
	#pragma scop
	for (t=0; t<TSTEPS-1; t++) {
		for (i=1; i<=N-2; i++) {
			for (j=1; j<=N-2; j++) {
				A[i][j] = (A[i-1][j-1] + A[i-1][ j ] + A[i-1][j+1]
				         + A[ i ][j-1] + A[ i ][ j ] + A[ i ][j+1]
				         + A[i+1][j-1] + A[i+1][ j ] + A[i+1][j+1])/9.0;
			}
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	A[8][9] = A[8][9];
	return 0;
}
