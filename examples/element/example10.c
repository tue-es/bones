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
// Filename...........element/example10.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#include <stdlib.h>

// This is 'example10', demonstrating multiple loops not starting at 0 for a 2D array and a kernel without a given name
int main(void) {
	int i,j;
	int N = 4;
	int M = 5;
	
	// Declare input/output arrays
	int A[N][M];
	int B[N][M];
	
	// Set the input data
	for(i=0;i<N;i++) {
		for(j=0;j<M;j++) {
			A[i][j] = i*M+j;
			B[i][j] = 9;
		}
	}
	
	// Perform the computation
	#pragma species kernel 2:N-1,1:M-1|element -> 2:N-1,1:M-1|element
	for(i=2;i<N;i++) {
		for(j=1;j<M;j++) {
			B[i][j] = A[i][j];
		}
	}
	#pragma species endkernel
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

