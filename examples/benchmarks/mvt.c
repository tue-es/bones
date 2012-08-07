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
// Filename...........benchmark/mvt.c
// Author.............Cedric Nugteren
// Last modified on...23-May-2012
//

#include "common.h"

// This is 'mvt', a matrix vector product and transpose kernel
int main(void) {
	int i,j;
	
	// Declare arrays on the stack
	float A[NX][NX];
	float x1[NX];
	float x2[NX];
	float y_1[NX];
	float y_2[NX];
	
	// Set the input data
	for (i=0; i<NX; i++) {
		x1[i] = ((float) i) / NX;
		x2[i] = ((float) i + 1) / NX;
		y_1[i] = ((float) i + 3) / NX;
		y_2[i] = ((float) i + 4) / NX;
		for (j=0; j<NX; j++) {
			A[i][j] = ((float) i*j) / NX;
		}
	}
	
	// Perform the computation
	#pragma species kernel 0:NX-1|element ^ 0:NX-1,0:NX-1|chunk(0:0,0:NX-1) ^ 0:NX-1|full -> 0:NX-1|element
	for (i=0; i<NX; i++) {
		for (j=0; j<NX; j++) {
			x1[i] = x1[i] + A[i][j] * y_1[j];
		}
	}
	#pragma species endkernel mvt-part1
	#pragma species kernel 0:NX-1|element ^ 0:NX-1,0:NX-1|chunk(0:NX-1,0:0) ^ 0:NX-1|full -> 0:NX-1|element
	for (i=0; i<NX; i++) {
		for (j=0; j<NX; j++) {
			x2[i] = x2[i] + A[j][i] * y_2[j];
		}
	}
	#pragma species endkernel mvt-part2
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

