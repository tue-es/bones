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
// Filename...........neighbourhood/example03.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define A 1024
#define B 1536

// Forward declarations of helper functions for statically allocated 2D memory
float ** alloc_2D(int size1, int size2);
void free_2D(float ** array_2D);

// This is 'example03', demonstrating a neighbourhood with only some values used (a cross) and a math.h square root function call
int main(void) {
	int i,j;
	int sizea = A;
	int sizeb = B;
	
	// Declare input/output arrays
	float **in = alloc_2D(sizea,sizeb);
	float **out = alloc_2D(sizea,sizeb);
	
	// Set the input data
	for(i=0;i<sizea;i++) {
		for(j=0;j<sizeb;j++) {
			in[i][j] = i+j;
		}
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel in[0:sizea-1,0:sizeb-1]|neighbourhood(-1:1,-1:1) -> out[0:sizea-1,0:sizeb-1]|element
	for(i=0;i<sizea;i++) {
		for(j=0;j<sizeb;j++) {
			if (i >= 1 && j >= 1 && i < (sizea-1) && j < (sizeb-1)) {
				out[i][j] =                 in[i+1][ j ] + 
				             in[ i ][j+1] + in[ i ][ j ] + in[ i ][j-1] +
				                            in[i-1][ j ]                 ;
			}
			else {
				out[i][j] = sqrt(in[i][j]);
			}
		}
	}
	#pragma species endkernel example3
	#pragma endscop
	
	// Clean-up and exit the function
	free_2D(in);
	free_2D(out);
	fflush(stdout);
	return 0;
}

// Helper function to allocate a 2D-array
float ** alloc_2D(int size1, int size2) {
	int a;
	float ** array_2D = (float **)malloc(size1*sizeof(float*));
	float * array_1D = (float *)malloc(size1*size2*sizeof(float));
	for (a=0; a<size1; a++) {
		array_2D[a] = &array_1D[a*size2];
	}
	return array_2D;
}

// Helper function to free a 2D-array
void free_2D(float ** array_2D) {
	free(array_2D[0]);
	free(array_2D);
}

