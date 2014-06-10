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
// Filename...........benchmark/fdtd-2d-apml.c
// Author.............Cedric Nugteren
// Last modified on...30-May-2012
//

#include "common.h"

// This is 'fdtd-2d-apml', a 2D finite different time domain kernel using an anisotropic perfectly matched layer
int main(void) {
	int i,j,k;
	int ix,iy,iz;
	float clf, tmp;
	float czm_iz, czp_iz;
	float cymh_iy, cyph_iy;
	
	// Declare arrays on the stack
	float czm[CZ+1];
	float czp[CZ+1];
	float cxmh[CXM+1];
	float cxph[CXM+1];
	float cymh[CYM+1];
	float cyph[CYM+1];
	float Ry[CZ+1][CYM+1];
	float Ax[CZ+1][CYM+1];
	float Ex[CZ+1][CYM+1][CXM+1];
	float Ey[CZ+1][CYM+1][CXM+1];
	float Hz[CZ+1][CYM+1][CXM+1];
	float Bza[CZ+1][CYM+1][CXM+1];
	
	// Set the constants
	float mui = 2341;
	float ch = 42;
	
	// Set the input data
	for (i=0; i<=CZ; i++) {
		czm[i] = ((float) i+1) / CXM;
		czp[i] = ((float) i+2) / CXM;
	}
	for (i=0; i<=CXM; i++) {
		cxmh[i] = ((float) i+3) / CXM;
		cxph[i] = ((float) i+4) / CXM;
	}
	for (i=0; i<=CYM; i++) {
		cymh[i] = ((float) i+5) / CXM;
		cyph[i] = ((float) i+6) / CXM;
	}
	for (i=0; i<=CZ; i++) {
		for (j=0; j<=CYM; j++) {
			Ry[i][j] = ((float) i*(j+1) + 10) / CYM;
			Ax[i][j] = ((float) i*(j+2) + 11) / CYM;
			for (k=0; k<=CXM; k++) {
				Ex[i][j][k] = ((float) i*(j+3) + k+1) / CXM;
				Ey[i][j][k] = ((float) i*(j+4) + k+2) / CYM;
				Hz[i][j][k] = ((float) i*(j+5) + k+3) / CZ;
			}
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		for (iz = 0; iz < CZ; iz++) {
			for (iy = 0; iy < CYM; iy++) {
				czm_iz = czm[iz];
				czp_iz = czp[iz];
				cymh_iy = cymh[iy];
				cyph_iy = cyph[iy];
				for (ix = 0; ix < CXM; ix++) {
					clf = Ex[iz][iy][ix] - Ex[iz][iy + 1][ix] + Ey[iz][iy][ix + 1] - Ey[iz][iy][ix];
					tmp = cymh_iy / cyph_iy * Bza[iz][iy][ix] - ch / cyph_iy * clf;
					Hz[iz][iy][ix] = Hz[iz][iy][ix] * (cxmh[ix] / cxph[ix]) + mui * czp_iz / cxph[ix] * tmp - mui * czm_iz / cxph[ix] * Bza[iz][iy][ix];
					Bza[iz][iy][ix] = tmp;
				}
				clf = Ex[iz][iy][CXM] - Ex[iz][iy + 1][CXM] + Ry[iz][iy] - Ey[iz][iy][CXM];
				tmp = cymh_iy / cyph_iy * Bza[iz][iy][CXM] - ch / cyph_iy * clf;
				Hz[iz][iy][CXM] = cxmh[CXM] / cxph[CXM] * Hz[iz][iy][CXM] + mui * czp_iz / cxph[CXM] * tmp - mui * czm_iz / cxph[CXM] * Bza[iz][iy][CXM];
				Bza[iz][iy][CXM] = tmp;
				for (ix = 0; ix < CXM; ix++) {
					clf = Ex[iz][CYM][ix] - Ax[iz][ix] + Ey[iz][CYM][ix + 1] - Ey[iz][CYM][ix];
					tmp = cymh[CYM] / cyph_iy * Bza[iz][iy][ix] - ch / cyph_iy * clf;
					Hz[iz][CYM][ix] = cxmh[ix] / cxph[ix] * Hz[iz][CYM][ix] + mui * czp_iz / cxph[ix] * tmp - mui * czm_iz / cxph[ix] * Bza[iz][CYM][ix];
					Bza[iz][CYM][ix] = tmp;
				}
				clf = Ex[iz][CYM][CXM] - Ax[iz][CXM] + Ry[iz][CYM] - Ey[iz][CYM][CXM];
				tmp = cymh[CYM] / cyph[CYM] * Bza[iz][CYM][CXM] - ch / cyph[CYM] * clf;
				Hz[iz][CYM][CXM] = cxmh[CXM] / cxph[CXM] * Hz[iz][CYM][CXM] + mui * czp_iz / cxph[CXM] * tmp - mui * czm_iz / cxph[CXM] * Bza[iz][CYM][CXM];
				Bza[iz][CYM][CXM] = tmp;
			}
		}
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

