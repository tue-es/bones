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
// Filename...........benchmark/dynprog.c
// Author.............Cedric Nugteren
// Last modified on...29-May-2012
//

#include "common.h"

// This is 'dynprog', a 2D dynamic programming algorithm
int main(void) {
	int i,j,k,iter;
	
	// Declare arrays on the stack
	float C[LENGTH][LENGTH];
	float W[LENGTH][LENGTH];
	float c[LENGTH][LENGTH];
	float sum_c[LENGTH][LENGTH][LENGTH];
	float out;
	
	// Set the input data
	for (i=0; i<LENGTH; i++) {
		for (j=0; j<LENGTH; j++) {
			C[i][j] = i*j % 2;
			W[i][j] = ((float) i-j) / LENGTH;
		}
	}
	
	// Perform the computation
	for (iter=0; iter<ITER; iter++) {
		#pragma species kernel 0:0|void -> 0:LENGTH-1,0:LENGTH-1|element
		for (i=0; i<=LENGTH-1; i++) {
			for (j=0; j<=LENGTH-1; j++) {
				c[i][j] = 0;
			}
		}
		#pragma species endkernel dynprog
		for (i=0; i<=LENGTH-2; i++) {
			for (j=i+1; j<=LENGTH-1; j++) {
				sum_c[i][j][i] = 0;
				for (k=i+1; k<=j-1; k++) {
					sum_c[i][j][k] = sum_c[i][j][k - 1] + c[i][k] + c[k][j];
				}
				c[i][j] = sum_c[i][j][j-1] + W[i][j];
			}
		}
		out += c[0][LENGTH-1];
	}
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

