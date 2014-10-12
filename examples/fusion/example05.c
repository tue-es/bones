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
// Filename...........fusion/example05.c
// Author.............Cedric Nugteren
// Last modified on...11-October-2014
//

#include <stdio.h>

// This is 'example05', like example02 but with constant values.
int main(void) {
	int i,j;
	
	// Declare input/output arrays
	int A[2048][1024];
	int B[2048][1024];
	int C[2048][1024];
	
	// Set the input data
	for(i=0;i<2048;i++) {
		for(j=0;j<1024;j++) {
			A[i][j] = i+j;
		}
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel A[0:2047,0:1023]|element -> B[0:2047,0:1023]|element
	for(i=0;i<2048;i++) {
		for(j=0;j<1024;j++) {
			B[i][j] = A[i][j] + 3;
		}
	}
	#pragma species endkernel example05-part1
	#pragma species kernel A[0:2047,0:979]|element -> C[0:2047,0:979]|element
	for(i=0;i<2048;i++) {
		for(j=0;j<980;j++) {
			C[i][j] = 9*A[i][j];
		}
	}
	#pragma species endkernel example05-part2
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	C[8][9] = C[8][9];
	return 0;
}

