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
// Filename...........chunk/example3.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#define BASE 2048
#define TILE 64
#define SIZE (BASE*TILE)

// This is 'example3', demonstrating a chunked input and a chunked output, and showing the importance of ordering (array referenced first should be placed first)
int main(void) {
	int i;
	int t = 0;
	float result = 0;
	
	// Declare input/output arrays
	float A[BASE];
	float B[SIZE];
	float out1[SIZE];
	float out2[SIZE];
	
	// Set the input data
	for(i=0;i<BASE;i++) {
		A[i] = 0.6;
	}
	for(i=0;i<SIZE;i++) {
		B[i] = i%6+i;
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:BASE-1]|element ^ B[0:(BASE*TILE)-1]|chunk(0:TILE-1) -> out1[0:(BASE*TILE)-1]|chunk(0:TILE-1) ^ out2[0:(BASE*TILE)-1]|chunk(0:TILE-1)
		for (i = 0; i < BASE; i++) {
			result = A[i];
			for (t = 0; t < TILE; t++) {
				result = result + t * B[i * TILE + t];
			}
			for (t = 0; t < TILE; t++) {
				out1[i * TILE + t] = result;
				out2[i * TILE + t] = -result;
			}
		}
		#pragma species endkernel example03_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

