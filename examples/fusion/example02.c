//
// This file is part of the Bones source-to-source compiler examples. This C-code
// example is meant to illustrate the use of Bones. For more information on Bones
// use the contact information below.
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........fusion/example02.c
// Author.............Cedric Nugteren
// Last modified on...09-July-2013
//

#include <stdio.h>
#define N 2048
#define M 512
// Condition: M must be smaller than N

// This is 'example02', an example of scalar kernel fusion with mismatching bounds but independent loop bodies.
int main(void) {
	int i,j;
	
	// Declare input/output arrays
	int A[N][M];
	int B[N][M];
	int C[N][M];
	
	// Set the input data
	for(i=0;i<N;i++) {
		for(j=0;j<M;j++) {
			A[i][j] = i+j;
		}
	}
	
	// Perform the computation
	#pragma species kernel A[0:N-1,10:M-1]|element -> B[0:N-1,10:M-1]|element
	for(i=0;i<N;i++) {
		for(j=10;j<M;j++) {
			B[i][j] = A[i][j] + 3;
		}
	}
	#pragma species endkernel example02-part1
	#pragma species kernel A[0:M-1,0:M-1]|element -> C[0:M-1,0:M-1]|element
	for(i=0;i<M;i++) {
		for(j=0;j<M;j++) {
			C[i][j] = -9*A[i][j];
		}
	}
	#pragma species endkernel example02-part2
	
	/*
	#pragma species kernel A[0:N-1,0:M-1]|element -> B[0:N-1,0:M-1]|element ^ C[0:N-1,0:M-1]|element
	for(i=0;i<MAX(N,M);i++) {
		for(j=0;j<M;j++) {
			if (j >= 10 && i < N) {
				B[i][j] = A[i][j] + 3;
			}
			if (i < M) {
				C[i][j] = -9*A[i][j];
			}
		}
	}
	#pragma species endkernel example02-fused
	*/
	
	// Clean-up and exit the function
	fflush(stdout);
	C[8][9] = C[8][9];
	return 0;
}

