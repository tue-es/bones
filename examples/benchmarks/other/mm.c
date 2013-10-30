//
// This file is part of the Bones source-to-source compiler examples. For more
// information on Bones please use the contact information below.
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........benchmark/mm.c
// Author.............Cedric Nugteren
// Last modified on...08-Jul-2013
//

#include <stdio.h>
#include <stdlib.h>
#define N 512

// This is 'mm', a matrix multiplication kernel
int main(void) {
	int i,j,k;
	
	// Declare arrays on the stack
	float A[N][N];
	float B[N][N];
	float C[N][N];
	
	// Set the input data
	for (i=0; i<N; i++) { for (k=0; k<N; k++) { A[i][k] = (i+k)*1.4; } }
	for (k=0; k<N; k++) { for (j=0; j<N; j++) { B[k][j] = k*j/0.9;   } }
	for (i=0; i<N; i++) { for (j=0; j<N; j++) { C[i][j] = 0;         } }
	
	// Perform the computation (C := A*B)
	#pragma species kernel 0:N-1,0:N-1|chunk(0:0,0:N-1) ^ 0:N-1,0:N-1|chunk(0:N-1,0:0) -> 0:N-1,0:N-1|element
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			C[i][j] = 0;
			for (k=0; k<N; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	#pragma species endkernel mm
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

