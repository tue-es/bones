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
// Filename...........fusion/example03.c
// Author.............Cedric Nugteren
// Last modified on...02-Oct-2013
//

#include <stdio.h>

// This is 'example03', with code similar to PolyBench's "2mm" benchmark. This is an example where fusion is only legal w.r.t. the i-loop
int main(void) {
	int i,j,k;
	
	// Declare arrays on the stack
	float A[2048][2048];
	float B[2048][2048];
	float C[2048][2048];
	float D[2048][2048];
	float tmp[2048][2048];
	
	// Set the constants
	int alpha = 32412;
	int beta = 2123;
	
	// Set the input data
	for (i=0; i<2048; i++) { for (j=0; j<2048; j++) { A[i][j] = ((float) i*j) / 2048; } }
	for (i=0; i<2048; i++) { for (j=0; j<2048; j++) { B[i][j] = ((float) i*(j+1)) / 2048; } }
	for (i=0; i<2048; i++) { for (j=0; j<2048; j++) { C[i][j] = ((float) i*(j+3)) / 2048; } }
	for (i=0; i<2048; i++) { for (j=0; j<2048; j++) { D[i][j] = ((float) i*(j+2)) / 2048; } }
	
	// Perform the computation (E := alpha*A*B*C + beta*D)
	#pragma species copyin A[0:2047,0:2047]|0 ^ B[0:2047,0:2047]|0 ^ D[0:2047,0:2047]|1 ^ C[0:2047,0:2047]|1
	#pragma species sync 0
	#pragma species kernel A[0:2047,0:2047]|chunk(0:0,0:2047) ^ B[0:2047,0:2047]|chunk(0:2047,0:0) -> tmp[0:2047,0:2047]|element
	for (i=0; i<2048; i++) {
		for (j=0; j<2048; j++) {
			tmp[i][j] = 0;
			for (k=0; k<2048; k++) {
				tmp[i][j] += alpha * A[i][k] * B[k][j];
			}
		}
	}
	#pragma species endkernel example03-part1
	#pragma species copyout tmp[0:2047,0:2047]|2
	#pragma species sync 1
	#pragma species kernel D[0:2047,0:2047]|element ^ tmp[0:2047,0:2047]|chunk(0:0,0:2047) ^ C[0:2047,0:2047]|chunk(0:2047,0:0) -> D[0:2047,0:2047]|element
	for (i=0; i<2048; i++) {
		for (j=0; j<2048; j++) {
			D[i][j] *= beta;
			for (k=0; k<2048; k++) {
				D[i][j] += tmp[i][k] * C[k][j];
			}
		}
	}
	#pragma species endkernel example03-part2
	#pragma species copyout D[0:2047,0:2047]|2
	#pragma species sync 2
	
	// Clean-up and exit the function
	fflush(stdout);
	D[8][9] = D[8][9];
	return 0;
}

