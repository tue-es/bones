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
// Filename...........dependences/example04.c
// Author.............Cedric Nugteren
// Last modified on...07-May-2013
//

#include <stdio.h>

// This is 'example04', a 2D example WITHOUT dependences.
int main(void) {
	int i,j;
	int N = 256;
	
	// Declare input/output arrays
	int D[N][N];
	
	// Set the input data
	for(i=0;i<N;i++) {
		for(j=0;j<N;j++) {
			D[i][j] = i*j+3;
		}
	}
	
	// Perform the computation
	#pragma species kernel D[1:8*N-7,1:8*N-7]|element -> D[6:2*N+4,6:2*N+4]|element
	for (i=0; i<N; i++) { // No dependence
		D[2*i+6][2*i+6] = D[8*i+1][8*i+1];
	}
	#pragma species endkernel example04
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

