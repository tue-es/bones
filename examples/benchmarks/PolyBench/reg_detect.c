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
// Filename...........benchmark/reg_detect.c
// Author.............Cedric Nugteren
// Last modified on...26-Jun-2012
//

#include "common.h"

// This is 'reg_detect', a regularity detection algorithm
int main(void) {
	int i,j,t,cnt;
	float sum;
	
	// Declare arrays on the stack
	float sum_tang[MAXGRID][MAXGRID];
	float mean[MAXGRID][MAXGRID];
	float path[MAXGRID][MAXGRID];
	float diff[MAXGRID][MAXGRID][LENGTH];
	float sum_diff[MAXGRID][MAXGRID][LENGTH];
	
	// Set the input data
	for (i=0; i<MAXGRID; i++) {
		for (j=0; j<MAXGRID; j++) {
			sum_tang[i][j] = (float)((i+1)*(j+1));
			mean[i][j] = ((float) i-j) / MAXGRID;
			path[i][j] = ((float) i*(j-1)) / MAXGRID;
		}
	}
	
	// Perform the computation
	#pragma scop
	for (t=0; t<ITER; t++) {
		#pragma species kernel 0:MAXGRID-1,0:MAXGRID-1|element -> 0:MAXGRID-1,0:MAXGRID-1,0:LENGTH-1|chunk(0:0,0:0,0:LENGTH-1)
		for (j=0; j<=MAXGRID-1; j++) {
			for (i=0; i<=MAXGRID-1; i++) {
				sum = sum_tang[j][i];
				for (cnt=0; cnt<=LENGTH-1; cnt++) {
					diff[j][i][cnt] = sum;
				}
			}
		}
		#pragma species endkernel reg-detect-part1
		for (j=0; j<=MAXGRID-1; j++) {
			for (i=j; i<=MAXGRID-1; i++) {
				sum_diff[j][i][0] = diff[j][i][0];
				for (cnt=1; cnt<=LENGTH-1; cnt++) {
					sum_diff[j][i][cnt] = sum_diff[j][i][cnt-1] + diff[j][i][cnt];
				}
				mean[j][i] = sum_diff[j][i][LENGTH-1];
			}
		}
		#pragma species kernel 0:0,0:MAXGRID-1|element -> 0:0,0:MAXGRID-1|element
		for (i=0; i<=MAXGRID-1; i++) {
			path[0][i] = mean[0][i];
		}
		#pragma species endkernel reg-detect-part2
		for (j=1; j<=MAXGRID-1; j++) {
			#pragma species kernel j-1:j-1,j-1:MAXGRID-2|element ^ j:j,j:MAXGRID-1|element -> j:j,j:MAXGRID-1|element
			for (i=j; i<=MAXGRID-1; i++) {
				path[j][i] = path[j-1][i-1] + mean[j][i];
			}
			#pragma species endkernel reg-detect-part3
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	path[8][9] = path[8][9];
	return 0;
}
