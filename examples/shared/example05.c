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
// Filename...........shared/example5.c
// Author.............Cedric Nugteren
// Last modified on...07-May-2012
//

#include <stdio.h>

// This is 'example5', demonstrating an inner-loop only classification of a reduction to scalar
int main(void) {
	int a,b,c;
	
	// Declare input/output arrays
	float in[16][16];
	float out[1];
	
	// Set the input data
	out[0] = -1;
	for(a=0;a<16;a++) {
		for(b=0;b<16;b++) {
			in[a][b] = 1.001;
		}
	}
	
	// Perform the computation
	for(a=0;a<16;a++) {
		#pragma species kernel a:a,0:a|element -> 0:0|shared
		for(b=0;b<=a;b++) {
			out[0] = out[0] - in[a][b]*in[a][b];
		}
		#pragma species endkernel example5
		out[0] = 1.002;
	}
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}
