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
// Filename...........benchmark/fdtd-2d.c
// Author.............Cedric Nugteren
// Last modified on...08-May-2012
//

#include "common.h"

// This is 'fdtd-2d', a 2D finite different time domain kernel
int main(void) {
	int i,j,t;
	
	// Declare arrays on the stack
	float ex[NI][NJ];
	float ey[NI][NJ];
	float hz[NI][NJ];
	
	// Set the input data
	for (i=0; i<NI; i++) {
		for (j=0; j<NJ; j++) {
			ex[i][j] = ((float) i*(j+1)) / NI;
			ey[i][j] = ((float) i*(j+2)) / NJ;
			hz[i][j] = ((float) i*(j+3)) / NI;
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		for (t = 0; t < TSTEPS; t++) {
			#pragma species kernel 0:0|void -> ey[0:0,0:NJ-1]|element
			for (j = 0; j < NJ; j++) {
				ey[0][j] = t;
			}
			#pragma species endkernel fdtd-2d_k2
			#pragma species kernel ey[1:NI-1,0:NJ-1]|element ^ hz[0:NI-1,0:NJ-1]|neighbourhood(-1:0,0:0) -> ey[1:NI-1,0:NJ-1]|element
			for (i = 1; i < NI; i++) {
				for (j = 0; j < NJ; j++) {
					ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
				}
			}
			#pragma species endkernel fdtd-2d_k3
			#pragma species kernel ex[0:NI-1,1:NJ-1]|element ^ hz[0:NI-1,0:NJ-1]|neighbourhood(0:0,-1:0) -> ex[0:NI-1,1:NJ-1]|element
			for (i = 0; i < NI; i++) {
				for (j = 1; j < NJ; j++) {
					ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
				}
			}
			#pragma species endkernel fdtd-2d_k4
			#pragma species kernel hz[0:NI-2,0:NJ-2]|element ^ ex[0:NI-2,0:NJ-1]|neighbourhood(0:0,0:1) ^ ey[0:NI-1,0:NJ-2]|neighbourhood(0:1,0:0) -> hz[0:NI-2,0:NJ-2]|element
			for (i = 0; i < NI - 1; i++) {
				for (j = 0; j < NJ - 1; j++) {
					hz[i][j] = hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
				}
			}
			#pragma species endkernel fdtd-2d_k5
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}
