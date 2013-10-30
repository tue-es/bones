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
// Filename...........benchmark/adi.c
// Author.............Cedric Nugteren
// Last modified on...29-May-2012
//

#include "common.h"

// This is 'adi', an alternating direction implicit solver
int main(void) {
	int t,i,j,i1,i2;
	
	// Declare arrays on the stack
	float X[N][N];
	float A[N][N];
	float B[N][N];
	
	// Set the input data
/*	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			X[i][j] = ((float) i*(j+1) + 1) / N;
			A[i][j] = ((float) i*(j+2) + 2) / N;
			B[i][j] = ((float) i*(j+3) + 3) / N;
		}
	}
*/	
	// Perform the computation
	#pragma scop
	for (t=0; t<TSTEPS; t++) {
		for (i1=0; i1<N; i1++) {
			for (i2=1; i2<N; i2++) {
				X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
				B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
			}
		}
		#pragma species kernel 0:N-1,N-1:N-1|element ^ 0:N-1,N-1:N-1|element -> 0:N-1,N-1:N-1|element
		for (i1=0; i1<N; i1++) {
			X[i1][N-1] = X[i1][N-1] / B[i1][N-1];
		}
		#pragma species endkernel adi-part1
		for (i1=0; i1<N; i1++) {
			for (i2=0; i2<N-2; i2++) {
				X[i1][N-i2-2] = (X[i1][N-2-i2] - X[i1][N-2-i2-1] * A[i1][N-i2-3]) / B[i1][N-3-i2];
			}
		}
		for (i1=1; i1<N; i1++) {
			for (i2=0; i2<N; i2++) {
				X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
				B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
			}
		}
		#pragma species kernel N-1:N-1,0:N-1|element  ^ N-1:N-1,0:N-1|element -> N-1:N-1,0:N-1|element
		for (i2=0; i2<N; i2++) {
			X[N-1][i2] = X[N-1][i2] / B[N-1][i2];
		}
		#pragma species endkernel adi-part2
		for (i1=0; i1<N-2; i1++) {
			for (i2=0; i2<N; i2++) {
				X[N-2-i1][i2] = (X[N-2-i1][i2] - X[N-i1-3][i2] * A[N-3-i1][i2]) / B[N-2-i1][i2];
			}
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	X[8][9] = X[8][9];
	return 0;
}

