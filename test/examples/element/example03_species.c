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
// Filename...........element/example3.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#define N1 2//8
#define N2 4//16
#define N3 8//32
#define N4 16//64

// This is 'example3', demonstrating a 4D array and defines for array sizes
int main(void) {
	int i,j,k,l;
	
	// Declare input/output arrays
	int A[N1][N2][N3][N4];
	int B[N1][N2][N3][N4];
	
	// Set the input data
	for(i=0;i<N1;i++) {
		for(j=0;j<N2;j++) {
			for(k=0;k<N3;k++) {
				for(l=0;l<N4;l++) {
					A[i][j][k][l] = i*j+k-l;
				}
			}
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:N1-1,0:N2-1,0:N3-1,0:N4-1]|element -> B[0:N1-1,0:N2-1,0:N3-1,0:N4-1]|element
		for (i = 0; i < N1; i++) {
			for (j = 0; j < N2; j++) {
				for (k = 0; k < N3; k++) {
					for (l = 0; l < N4; l++) {
						B[i][j][k][l] = 3 * A[i][j][k][l] + 6;
					}
				}
			}
		}
		#pragma species endkernel example03_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

