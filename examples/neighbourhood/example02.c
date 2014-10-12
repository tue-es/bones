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
// Filename...........neighbourhood/example02.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>
#define A 256
#define B 512

// This is 'example02', demonstrating a 2D array, a 2D neighbourhood and a for-loop-less notation of the neighbourhood accesses
int main(void) {
	int i,j;
	
	// Declare input/output arrays
	float in[A][B];
	float out[A][B];
	
	// Set the input data
	for(i=0;i<A;i++) {
		for(j=0;j<B;j++) {
			in[i][j] = i+j;
		}
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel in[0:255,0:511]|neighbourhood(-1:1,-1:1) -> out[0:255,0:511]|element
	for(i=0;i<A;i++) {
		for(j=0;j<B;j++) {
			if (i >= 1 && j >= 1 && i < (A-1) && j < (B-1)) {
				out[i][j] = (in[i+1][j+1] + in[i+1][ j ] + in[i+1][j-1] +
				             in[ i ][j+1] + in[ i ][ j ] + in[ i ][j-1] +
				             in[i-1][j+1] + in[i-1][ j ] + in[i-1][j-1])/9.0;
			}
			else {
				out[i][j] = in[i][j];
			}
		}
	}
	#pragma species endkernel example2
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

