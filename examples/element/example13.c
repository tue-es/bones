//
// This file is part of the Bones source-to-source compiler examples. This C-code
// example is meant to illustrate the use of Bones. For more information on Bones
// use the contact information below.
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........element/example13.c
// Author.............Cedric Nugteren
// Last modified on...07-May-2013
//

#include <stdio.h>

// This is 'example13', an example with multiple loop nests and various if-statements
int main(void) {
	int i,j;
	int N = 256;
	
	// Declare input/output arrays
	int A[N];
	int B[N];
	int C[N];
	int D[N][N];
	int E[N][N];
	
	// Set the input data
	for(i=0;i<N;i++) {
		A[i] = i;
		B[i] = i+5;
		C[i] = i+9;
		for(j=0;j<N;j++) {
			D[i][j] = i*j+3;
			E[i][j] = i*j+9;
		}
	}
	
	// Perform the computation
	#pragma species kernel C[0:N-1]|element -> B[11:N-1]|element ^ A[0:5]|element
	for (i=0; i<N; i++) {
		if (i > 10) {
			B[i] = C[i];
		}
		if (i < 6) {
			A[i] = C[i];
		}
	}
	#pragma species endkernel example13_k1
	#pragma species kernel A[50:N-1]|element -> B[50:N-1]|element
	for (i=0; i<N-9; i++) {
		if (i+10 > 50) {
			B[i+9] = A[i+9];
		}
	}
	#pragma species endkernel example13_k2
	#pragma species kernel E[5:N-1,0:N-1]|element -> D[5:N-1,0:N-1]|element
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			if (i > 4) {
				D[i][j] = E[i][j];
			}
		}
	}
	#pragma species endkernel example13_k3
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

