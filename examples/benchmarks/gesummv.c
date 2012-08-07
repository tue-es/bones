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
// Filename...........benchmark/gesummv.c
// Author.............Cedric Nugteren
// Last modified on...20-Jul-2012
//

#include "common.h"

// This is 'gesummv', a general scalar, vector and matrix multiplication kernel
int main(void) {
	int i,j;
	
	// Declare arrays on the stack
	float A[NX][NX];
	float B[NX][NX];
	float x[NX];
	float y[NX];
	float tmp[NX];
	
	// Set the constants
	float alpha = 43532;
	float beta = 12313;
	
	// Set the input data
	for (i=0; i<NX; i++) {
		x[i] = ((float) i) / NX;
		for (j=0; j<NX; j++) {
			A[i][j] = ((float) i*(j+1)) / NX;
			B[i][j] = ((float) (i+3)*j) / NX;
		}
	}
	
	// Perform the computation
	#pragma species kernel 0:NX-1,0:NX-1|chunk(0:0,0:NX-1) ^ 0:NX-1|full ^ 0:NX-1,0:NX-1|chunk(0:0,0:NX-1) -> 0:NX-1|element ^ 0:NX-1|element
	for (i=0; i<NX; i++) {
		tmp[i] = 0;
		y[i] = 0;
		for (j=0; j<NX; j++) {
			tmp[i] = A[i][j] * x[j] + tmp[i];
			y[i] = B[i][j] * x[j] + y[i];
		}
		y[i] = alpha*tmp[i] + beta*y[i];
	}
	#pragma species endkernel gesummv
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

