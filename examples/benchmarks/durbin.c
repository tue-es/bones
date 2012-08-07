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
// Filename...........benchmark/durbin.c
// Author.............Cedric Nugteren
// Last modified on...03-Jul-2012
//

#include "common.h"

// This is 'durbin', an algorithm to solve an equation involving a Toeplitz matrix using the Levinson-Durbin recursion
int main(void) {
	int i,j,k;
	float alpha_k;
	
	// Declare arrays on the stack
	float alpha[NX];
	float beta[NX];
	float r[NX];
	float y[NX][NX];
	float sum[NX][NX];
	float out[NX];
	
	// Set the input data
	for (i=0; i<NX; i++) {
		alpha[i] = i;
		beta[i] = (i+1)/NX/2.0;
		r[i] = (i+1)/NX/4.0;
		for (j=0; j<NX; j++) {
			y[i][j] = ((float) i*j) / NX;
			sum[i][j] = ((float) i*j) / NX;
		}
	}
	
	// Perform the computation
	y[0][0] = r[0];
	beta[0] = 1;
	alpha[0] = r[0];
	for (k=1; k<NX; k++) {
		beta[k] = beta[k-1] - alpha[k-1] * alpha[k-1] * beta[k-1];
		sum[0][k] = r[k];
		for (i=0; i<=k-1; i++) {
			sum[i+1][k] = sum[i][k] + r[k-i-1] * y[i][k-1];
		}
		alpha[k] = -sum[k][k] * beta[k];
		alpha_k = alpha[k];
		#pragma species kernel 0:k-1,k-1:k-1|element ^ k-1:0,k-1:k-1|element -> 0:k-1,k:k|element
		for (i=0; i<=k-1; i++) {
			y[i][k] = y[i][k-1] + alpha_k * y[k-i-1][k-1];
		}
		#pragma species endkernel durbin-part1
		y[k][k] = alpha[k];
	}
	#pragma species kernel 0:NX-1,NX-1:NX-1|element -> 0:NX-1|element
	for (i=0; i<NX; i++) {
		out[i] = y[i][NX-1];
	}
	#pragma species endkernel durbin-part2
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

