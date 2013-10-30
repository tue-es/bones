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
// Filename...........element/example5.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#define SIZE 2311

// This is 'example5', demonstrating multiple inputs and outputs of different types
int main(void) {
	int i;
	float result = 0;
	
	// Declare input/output arrays
	float in1[SIZE];
	int in2[SIZE];
	char in3[SIZE];
	float out1[SIZE];
	float out2[SIZE];
	
	// Set the input data
	for(i=0;i<SIZE;i++) {
		in1[i] = -0.34*i;
		in2[i] = i*3;
		in3[i] = i%100;
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel in2[0:SIZE-1]|element ^ in1[0:SIZE-1]|element ^ in3[0:SIZE-1]|element -> out1[0:SIZE-1]|element ^ out2[0:SIZE-1]|element
		for (i = 0; i < SIZE; i++) {
			if (in3[i] > 50) {
				result = in2[i] / in1[i];
			} else {
				result = in2[i] * in1[i];
			}
			out1[i] = result;
			out2[i] = in3[i] / 255.0;
		}
		#pragma species endkernel example05_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

