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
// Filename...........benchmark/3mm.c
// Author.............Cedric Nugteren
// Last modified on...03-April-2012
//

#include "common.h"

// This is '3mm', a 3 matrix multiply kernel
int main(void) {
	int i,j,k;
	
	// Declare arrays on the stack
	float A[NI][NK];
	float B[NK][NJ];
	float C[NJ][NM];
	float D[NM][NL];
	float E[NI][NJ];
	float F[NJ][NL];
	float G[NI][NL];
	
	// Set the input data
	for (i=0; i<NI; i++) { for (j=0; j<NK; j++) { A[i][j] = ((float) i*j) / NI; } }
	for (i=0; i<NK; i++) { for (j=0; j<NJ; j++) { B[i][j] = ((float) i*(j+1)) / NJ; } }
	for (i=0; i<NL; i++) { for (j=0; j<NJ; j++) { C[i][j] = ((float) i*(j+3)) / NL; } }
	for (i=0; i<NI; i++) { for (j=0; j<NL; j++) { D[i][j] = ((float) i*(j+2)) / NK; } }
	
	// Perform the computation (G := E*F, with E := A*B and F := C*D)
	#pragma scop
	{
		#pragma species kernel A[0:NI-1,0:NK-1]|chunk(0:0,0:NK-1) ^ B[0:NK-1,0:NJ-1]|chunk(0:NK-1,0:0) -> E[0:NI-1,0:NJ-1]|element
		for (i = 0; i < NI; i++) {
			for (j = 0; j < NJ; j++) {
				E[i][j] = 0;
				for (k = 0; k < NK; k++) {
					E[i][j] += A[i][k] * B[k][j];
				}
			}
		}
		#pragma species endkernel 3mm_k1
		#pragma species kernel C[0:NJ-1,0:NM-1]|chunk(0:0,0:NM-1) ^ D[0:NM-1,0:NL-1]|chunk(0:NM-1,0:0) -> F[0:NJ-1,0:NL-1]|element
		for (i = 0; i < NJ; i++) {
			for (j = 0; j < NL; j++) {
				F[i][j] = 0;
				for (k = 0; k < NM; k++) {
					F[i][j] += C[i][k] * D[k][j];
				}
			}
		}
		#pragma species endkernel 3mm_k2
		#pragma species kernel E[0:NI-1,0:NJ-1]|chunk(0:0,0:NJ-1) ^ F[0:NJ-1,0:NL-1]|chunk(0:NJ-1,0:0) -> G[0:NI-1,0:NL-1]|element
		for (i = 0; i < NI; i++) {
			for (j = 0; j < NL; j++) {
				G[i][j] = 0;
				for (k = 0; k < NJ; k++) {
					G[i][j] += E[i][k] * F[k][j];
				}
			}
		}
		#pragma species endkernel 3mm_k3
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

