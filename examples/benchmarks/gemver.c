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
// Filename...........benchmark/gemver.c
// Author.............Cedric Nugteren
// Last modified on...04-April-2012
//

#include "common.h"

// This is 'gemver', a general matrix vector multiplication and matrix addition kernel
int main(void) {
	int i,j;
	
	// Declare arrays on the stack
	float A[NX][NX];
	float u1[NX];
	float u2[NX];
	float v1[NX];
	float v2[NX];
	float w[NX];
	float x[NX];
	float y[NX];
	float z[NX];
	
	// Set the constants
	int alpha = 43532;
	int beta = 12313;
	
	// Set the input data
	for (i=0; i<NX; i++) {
		u1[i] = i;
		u2[i] = (i+1)/NX/2.0;
		v1[i] = (i+1)/NX/4.0;
		v2[i] = (i+1)/NX/6.0;
		w[i] = 0.0;
		x[i] = 0.0;
		y[i] = (i+1)/NX/8.0;
		z[i] = (i+1)/NX/9.0;
		for (j=0; j<NX; j++) {
			A[i][j] = ((float) i*j) / NX;
		}
	}
	
	// Perform the computation
	#pragma species kernel 0:NX-1,0:NX-1|element ^ 0:NX-1|element ^ 0:NX-1|element ^ 0:NX-1|element ^ 0:NX-1|element -> 0:NX-1,0:NX-1|element
	for (i=0; i<NX; i++) {
		for (j=0; j<NX; j++) {
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}
	#pragma species endkernel gemver-part1
	#pragma species kernel 0:NX-1|element ^ 0:NX-1,0:NX-1|chunk(0:NX-1,0:0) ^ 0:NX-1|full -> 0:NX-1|element
	for (i=0; i<NX; i++) {
		for (j=0; j<NX; j++) {
			x[i] = x[i] + beta * A[j][i] * y[j];
		}
	}
	#pragma species endkernel gemver-part2
	#pragma species kernel 0:NX-1|element ^ 0:NX-1|element -> 0:NX-1|element
	for (i=0; i<NX; i++) {
		x[i] = x[i] + z[i];
	}
	#pragma species endkernel gemver-part3
	#pragma species kernel 0:NX-1|element ^ 0:NX-1,0:NX-1|chunk(0:0,0:NX-1) ^ 0:NX-1|full -> 0:NX-1|element
	for (i=0; i<NX; i++) {
		for (j=0; j<NX; j++) {
			w[i] = w[i] +  alpha * A[i][j] * x[j];
		}
	}
	#pragma species endkernel gemver-part4
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

