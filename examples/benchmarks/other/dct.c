//
// This file is part of the Bones source-to-source compiler examples. For more
// information on Bones please use the contact information below.
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........benchmark/dct.c
// Author.............Cedric Nugteren
// Last modified on...15-Jul-2013
//

#include <stdio.h>
#include <stdlib.h>

#define N 2048

// This is 'dct', a 2D 8x8 discrete cosine transform kernel
int main(void) {
	int i;
	int x,y;
	int u,v;
	
	// Declare arrays on the stack
	float A[N*N];
	float B[N*N];
	
	// Set the input data
	for (i=0; i<N*N; i++) {
		A[i] = i*1.4;
		B[i] = i/0.9;
	}
	
	// Set the constants
	float alpha = 1.414213f * 0.5f;
	
	// Perform the computation (y := ax+y)
	#pragma scop
	for (y=0; y<8; y++) {
		for (x=0; x<8; x++) {
			B[y*8+x] = 0;
			for (u=0; u<8; u++) {
				for (v=0; v<8; v++) {
					B[y*8+x] += alpha * alpha * A[u*8+v] *
					            cos(PI * u * (2.0f*x+1) * (1.0f/16.0f)) *
					            cos(PI * v * (2.0f*y+1) * (1.0f/16.0f));
				}
			}
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}