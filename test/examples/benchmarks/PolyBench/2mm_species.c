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
// Filename...........benchmark/2mm.c
// Author.............Cedric Nugteren
// Last modified on...03-April-2012
//

#include "common.h"

// This is '2mm', a 2 matrix multiply kernel
int main(void) {
	int i,j,k;
	
	// Declare arrays on the stack
	float A[NI][NK];
	float B[NK][NJ];
	float C[NJ][NL];
	float D[NI][NL];
	float tmp[NI][NJ];
	
	// Set the constants
	int alpha = 32412;
	int beta = 2123;
	
	// Set the input data
	for (i=0; i<NI; i++) { for (j=0; j<NK; j++) { A[i][j] = ((float) i*j) / NI; } }
	for (i=0; i<NK; i++) { for (j=0; j<NJ; j++) { B[i][j] = ((float) i*(j+1)) / NJ; } }
	for (i=0; i<NL; i++) { for (j=0; j<NJ; j++) { C[i][j] = ((float) i*(j+3)) / NL; } }
	for (i=0; i<NI; i++) { for (j=0; j<NL; j++) { D[i][j] = ((float) i*(j+2)) / NK; } }
	
	// Perform the computation (E := alpha*A*B*C + beta*D)
	#pragma scop
	{
		#pragma species kernel A[0:NI-1,0:NK-1]|chunk(0:0,0:NK-1) ^ B[0:NK-1,0:NJ-1]|chunk(0:NK-1,0:0) -> tmp[0:NI-1,0:NJ-1]|element
		for (i = 0; i < NI; i++) {
			for (j = 0; j < NJ; j++) {
				tmp[i][j] = 0;
				for (k = 0; k < NK; k++) {
					tmp[i][j] += alpha * A[i][k] * B[k][j];
				}
			}
		}
		#pragma species endkernel 2mm_k1
		#pragma species kernel D[0:NI-1,0:NL-1]|element ^ tmp[0:NI-1,0:NJ-1]|chunk(0:0,0:NJ-1) ^ C[0:NJ-1,0:NL-1]|chunk(0:NJ-1,0:0) -> D[0:NI-1,0:NL-1]|element
		for (i = 0; i < NI; i++) {
			for (j = 0; j < NL; j++) {
				D[i][j] *= beta;
				for (k = 0; k < NJ; k++) {
					D[i][j] += tmp[i][k] * C[k][j];
				}
			}
		}
		#pragma species endkernel 2mm_k2
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

