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
// Filename...........benchmark/atax.c
// Author.............Cedric Nugteren
// Last modified on...03-April-2012
//

#include "common.h"

// This is 'atax', a matrix transpose and vector multiplication kernel
int main(void) {
	int i,j;
	
	// Declare arrays on the stack
	float A[NX][NY];
	float x[NY];
	float y[NX];
	float tmp[NX];
	
	// Set the input data
	for (i=0; i<NY; i++) {
		x[i] = i*3.14159;
	}
	for (i=0; i<NX; i++) {
		for (j=0; j<NY; j++) {
			A[i][j] = ((float) i*(j+1)) / NX;
		}
	}
	
	// Perform the computation (y := A'Ax)
	#pragma species kernel 0:NX-1,0:NY-1|chunk(0:0,0:NY-1) ^ 0:NY-1|full -> 0:NX-1|element
	for (i=0; i<NX; i++) {
		tmp[i] = 0;
		for (j=0; j<NY; j++) {
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
	}
	#pragma species endkernel atax-part1
	#pragma species kernel 0:NX-1,0:NY-1|chunk(0:NX-1,0:0) ^ 0:NX-1|full -> 0:NY-1|element
	for (j=0; j<NY; j++) {
		y[j] = 0;
		for (i=0; i<NX; i++) {
			y[j] = y[j] + A[i][j] * tmp[i];
		}
	}
	#pragma species endkernel atax-part2
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

