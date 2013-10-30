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
// Filename...........benchmark/doitgen.c
// Author.............Cedric Nugteren
// Last modified on...10-April-2012
//

#include "common.h"

// This is 'doitgen', a multiresolution analysis kernel
int main(void) {
	int i,j,k,p,q,r,s;
	
	// Declare arrays on the stack
	float A[NR][NQ][NP];
	float sum[NR][NQ][NP];
	float C4[NP][NP];
	
	// Set the input data
	for (i=0; i<NR; i++) { for (j=0; j<NQ; j++) { for (k=0; k<NP; k++) { A[i][j][k] = ((float) i*j + k) / NP; } } }
	for (i=0; i<NP; i++) { for (j=0; j<NP; j++) { C4[i][j] = ((float) i*j) / NP; } }
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:NR-1,0:NQ-1,0:NP-1]|chunk(0:0,0:0,0:NP-1) ^ C4[0:NP-1,0:NP-1]|chunk(0:NP-1,0:0) -> sum[0:NR-1,0:NQ-1,0:NP-1]|element
		for (r = 0; r < NR; r++) {
			for (q = 0; q < NQ; q++) {
				for (p = 0; p < NP; p++) {
					sum[r][q][p] = 0;
					for (s = 0; s < NP; s++) {
						sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
					}
				}
			}
		}
		#pragma species endkernel doitgen_k1
		#pragma species kernel sum[0:NR-1,0:NQ-1,0:NP-1]|element -> A[0:NR-1,0:NQ-1,0:NP-1]|element
		for (r = 0; r < NR; r++) {
			for (q = 0; q < NQ; q++) {
				for (p = 0; p < NP; p++) {
					A[r][q][p] = sum[r][q][p];
				}
			}
		}
		#pragma species endkernel doitgen_k2
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

