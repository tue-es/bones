//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example application: 'nw', taken from
// the Rodinia benchmark suite. For more information on the application or on Bones
// please use the contact information below.
//
// == More information on Hotspot
// Original code......https://www.cs.virginia.edu/~skadron/wiki/rodinia/
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/nw.c
// Authors............Cedric Nugteren
// Last modified on...01-Jun-2014
//

//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//########################################################################
//### Defines
//########################################################################

// Config
#define MAX_ROWS (2048+1)
#define MAX_COLS (2048+1)
#define PENALTY 10

// Reference
int blosum62[24][24] = {
	{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
	{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
	{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
	{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
	{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
	{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
	{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
	{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
	{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
	{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
	{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
	{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
	{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
	{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
	{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
	{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
	{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
	{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
	{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
	{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {
	printf("[nw] Start Needleman-Wunsch\n");

	// Arrays
	int similarity[MAX_ROWS][MAX_COLS];
	int items[MAX_ROWS][MAX_COLS];

	// Initialize random input data
	srand (7);
	for (int i=0; i<MAX_ROWS; i++) {
		for (int j=0; j<MAX_COLS; j++) {
			items[i][j] = 0;
		}
	}
	for (int i=1; i<MAX_ROWS; i++) {
		items[i][0] = rand()%10 + 1;
	}
	for (int j=1; j<MAX_COLS; j++) {
		items[0][j] = rand()%10 + 1;
	}

	// Initialize reference
	for (int i=0; i<MAX_ROWS; i++) {
		for (int j=0; j<MAX_COLS; j++) {
			similarity[i][j] = blosum62[items[i][0]][items[0][j]];
		}
	}

	// Update input with penalty
	for (int i=1; i<MAX_ROWS; i++) {
		items[i][0] = -i*PENALTY;
	}
	for (int j=1; j<MAX_COLS; j++) {
		items[0][j] = -j*PENALTY;
	}

	// Start of computation
	#pragma scop

	// Compute top-left matrix 
	for (int i=0; i<MAX_ROWS-2; i++) {
		for (int idx=0; idx <= i; idx++) {
			int a = items[idx+0][i+0-idx] + similarity[idx+1][i+1-idx];
			int b = items[idx+1][i+0-idx] - PENALTY;
			int c = items[idx+0][i+1-idx] - PENALTY;
			int max_val = a;
			if (b > max_val) {
				max_val = b;
			}
			if (c > max_val) {
				max_val = c;
			}
			items[idx+1][i+1-idx] = max_val;
		}
	}

	// Compute bottom-right matrix 
	for (int i=MAX_ROWS-4; i>=0; i--) {
		for (int idx=0; idx <= i; idx++) {
			int a = items[MAX_ROWS-idx-3][idx+MAX_COLS-i-3] + similarity[MAX_ROWS-idx-2][idx+MAX_COLS-i-2];
			int b = items[MAX_ROWS-idx-2][idx+MAX_COLS-i-3] - PENALTY;
			int c = items[MAX_ROWS-idx-3][idx+MAX_COLS-i-2] - PENALTY;
			int max_val = a;
			if (b > max_val) {
				max_val = b;
			}
			if (c > max_val) {
				max_val = c;
			}
			items[MAX_ROWS-idx-2][idx+MAX_COLS-i-2] = max_val;
		}
	}

	// End of computation
	#pragma endscop

	// Clean-up and exit
	printf("\n[nw] Completed\n\n"); fflush(stdout);
	fflush(stdout);
	return 0;
}

//########################################################################