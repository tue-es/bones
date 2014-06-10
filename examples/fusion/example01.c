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
// Filename...........fusion/example01.c
// Author.............Cedric Nugteren
// Last modified on...09-July-2013
//

#include <stdio.h>
#define N 512
#define M 2048

// This is 'example01', a basic example of an opportunity for scalar kernel fusion.
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
	#pragma species kernel A[0:N-1,0:M-1]|element -> B[0:N-1,0:M-1]|element
	for(i=0;i<N;i++) {
		for(j=0;j<M;j++) {
			B[i][j] = 2*A[i][j];
		}
	}
	#pragma species endkernel example01-part1
	#pragma species kernel B[0:N-1,0:M-1]|element -> C[0:N-1,0:M-1]|element
	for(i=0;i<N;i++) {
		for(j=0;j<M;j++) {
			C[i][j] = 8*B[i][j];
		}
	}
	#pragma species endkernel example01-part2
	
	/*
	#pragma species kernel A[0:N-1,0:M-1]|element -> B[0:N-1,0:M-1]|element ^ C[0:N-1,0:M-1]|element
	for(i=0;i<N;i++) {
		for(j=0;j<M;j++) {
			B[i][j] = 2*A[i][j];
			C[i][j] = 8*B[i][j];
		}
	}
	#pragma species endkernel example01-fused
	*/
	
	// Clean-up and exit the function
	fflush(stdout);
	C[8][9] = C[8][9];
	return 0;
}

