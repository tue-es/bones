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
// Filename...........benchmark/lu.c
// Author.............Cedric Nugteren
// Last modified on...26-Jun-2012
//

#include "common.h"

// This is 'lu', an LU decomposition kernel
int main(void) {
	int i,j,k;
	
	// Declare arrays on the stack
	float A[N][N];
	
	// Set the input data
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			A[i][j] = ((float) (i+1)*(j+1)) / N;
		}
	}
	
	// Perform the computation
	#pragma scop
	for (k=0; k<N; k++) {
		#pragma species kernel k:k,k+1:N-1|element -> k:k,k+1:N-1|element
		for (j=k+1; j<N; j++) {
			A[k][j] = A[k][j] / A[k][k];
		}
		#pragma species endkernel lu-part1
		#pragma species kernel k+1:N-1,k:k|element ^ k:k,k+1:N-1|element ^ k+1:N-1,k+1:N-1|element -> k+1:N-1,k+1:N-1|element
		for(i=k+1; i<N; i++) {
			for (j=k+1; j<N; j++) {
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
		}
		#pragma species endkernel lu-part2
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	A[8][9] = A[8][9];
	return 0;
}

