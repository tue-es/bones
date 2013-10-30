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
// Filename...........benchmark/gramschmidt.c
// Author.............Cedric Nugteren
// Last modified on...26-Jun-2012
//

#include "common.h"

// This is 'gramschmidt', an algorithm for the Gram-Schmidt process
int main(void) {
	int i,j,k;
	float nrm[1];
	float rkk;
	float rkj;
	
	// Declare arrays on the stack
	float A[NI][NJ];
	float R[NJ][NJ];
	float Q[NI][NJ];
	
	// Set the input data
	for (i=0; i<NI; i++) {
		for (j=0; j<NJ; j++) {
			A[i][j] = ((float) i*j) / NI + 1;
			Q[i][j] = ((float) i*(j+1)) / NJ;
		}
	}
	for (i=0; i<NJ; i++) {
		for (j=0; j<NJ; j++) {
			R[i][j] = ((float) i*(j+2)) / NJ;
		}
	}
	
	// Perform the computation
	for (k=0; k<NJ; k++) {
		nrm[0] = 0;
		#pragma species kernel 0:NI-1,k:k|element -> 0:0|shared
		for (i=0; i<NI; i++) {
			nrm[0] += A[i][k] * A[i][k];
		}
		#pragma species endkernel gramschmidt-part1
		R[k][k] = sqrt(nrm[0]);
		rkk = R[k][k];
		#pragma species kernel 0:NI-1,k:k|element -> 0:NI-1,k:k|element
		for (i=0; i<NI; i++) {
			Q[i][k] = A[i][k] / rkk;
		}
		#pragma species endkernel gramschmidt-part2
		#pragma species kernel 0:NI-1,k:k|element ^ 0:NI-1,k+1:NJ-1|element -> 0:NI-1,k+1:NJ-1|element ^ k:k,k+1:NJ-1|element
		for (j=k+1; j<NJ; j++) {
			R[k][j] = 0;
			#pragma species kernel 0:NI-1,k:k|element ^ 0:NI-1,j:j|element -> k:k,j:j|shared
			for (i=0; i<NI; i++) {
				R[k][j] += Q[i][k] * A[i][j];
			}
			#pragma species endkernel gramschmidt-part3a
			rkj = R[k][j];
			#pragma species kernel 0:NI-1,j:j|element ^ 0:NI-1,k:k|element -> 0:NI-1,j:j|element 
			for (i=0; i<NI; i++) {
				A[i][j] = A[i][j] - Q[i][k] * rkj;
			}
			#pragma species endkernel gramschmidt-part3b
		}
		#pragma species endkernel gramschmidt-part3
	}
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}