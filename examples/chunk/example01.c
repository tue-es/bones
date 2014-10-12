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
// Filename...........chunk/example01.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>

// This is 'example01', a basic chunk to element example using a 2D tile
int main(void) {
	int i,j;
	int i2,j2;
	int result = 0;
	
	// Declare input/output arrays
	int A[560][32];
	int B[56][16];
	
	// Set the input data
	for(i=0;i<560;i++) {
		for(j=0;j<32;j++) {
			A[i][j] = i+j;
		}
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel A[0:559,0:31]|chunk(0:9,0:1) -> B[0:55,0:15]|element
	for(i=0;i<56;i++) {
		for(j=0;j<16;j++) {
			result = 0;
			for (i2=0;i2<10;i2++) {
				for (j2=0;j2<2;j2++) {
					result = result + A[i*10+i2][j*2+j2];
				}
			}
			B[i][j] = result;
		}
	}
	#pragma species endkernel example1
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

