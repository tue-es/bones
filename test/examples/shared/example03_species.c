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
// Filename...........shared/example3.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#include <stdlib.h>
#define SIZE 1024

// This is 'example3', demonstrating a reduction to a 2D array
int main(void) {
	int i,p,q;
	int index1,index2;
	
	// Declare input/output arrays
	int *in = (int *)malloc(SIZE*sizeof(int));
	int B[20][10];
	
	// Set the input data
	for(i=0;i<SIZE;i++) {
		in[i] = (SIZE-i);
	}
	
	// Set the output to zero before starting
	for(p=0;p<20;p++) {
		for(q=0;q<10;q++) {
			B[p][q] = 0;
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel in[0:SIZE-1]|element ^ B[0:19,0:9]|full -> B[0:19,0:9]|shared
		for (i = 0; i < SIZE; i++) {
			index1 = in[i] % 20;
			index2 = in[i] % 10;
			B[index1][index2] = B[index1][index2] + 1;
		}
		#pragma species endkernel example03_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	free(in);
	fflush(stdout);
	return 0;
}

