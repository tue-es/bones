//
// This file is part of the Bones source-to-source compiler examples. This header
// contains array size definitions and is common among the examples that are also
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
// Filename...........benchmark/polybench.h
// Author.............Cedric Nugteren
// Last modified on...23-May-2012
//

// Include C-libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Select a dataset size
//#define MINI_DATASET
#define SMALL_DATASET
//#define STANDARD_DATASET
//#define LARGE_DATASET
//#define EXTRALARGE_DATASET

// Defines used per benchmark:
//
// N        [adi, cholesky, correlation, covariance, floyd-warshall, jacobi-2d-imper, lu, ludcmp, seidel-2d]
// M        [correlation, covariance]
// NI       [2mm, 3mm, fdtd-2d, gemm, symm, syrk, syr2k, trmm]
// NJ       [2mm, 3mm, fdtd-2d, gemm, symm, syrk, syr2k]
// NK       [2mm, 3mm, gemm]
// NL       [2mm, 3mm]
// NQ       [doitgen]
// NR       [doitgen]
// NP       [doitgen]
// NM       [3mm]
// NX       [atax, bicg, durbin, gemver, gesummv, mvt, trisolv]
// NY       [atax, bicg]
// CZ       [fdtd-2d-apml]
// CYM      [fdtd-2d-apml]
// CXM      [fdtd-2d-apml]
// LARGE_N  [jacobi-1d-imper]
// LENGTH   [dynprog, reg_detect]
// TSTEPS   [adi, fdtd-2d, jacobi-1d-imper, jacobi-2d-imper, seidel-2d]
// ITER     [dynprog, reg_detect]
// MAXGRID  [reg_detect]
//

// Determine the sizes of the 5 possible datasets
#ifdef MINI_DATASET
	#define N 32
	#define M 32
	#define NI 32
	#define NJ 32
	#define NK 32
	#define NL 32
	#define NM 32
	#define NQ 10
	#define NR 10
	#define NP 10
	#define NX 32
	#define NY 32
	#define CZ 32
	#define CYM 32
	#define CXM 32
	#define LARGE_N 500
	#define LENGTH 32
	#define TSTEPS 2
	#define ITER 10
	#define MAXGRID 2
#endif
#ifdef SMALL_DATASET
	#define N 256
	#define M 256
	#define NI 128
	#define NJ 128
	#define NK 128
	#define NL 128
	#define NM 128
	#define NQ 32
	#define NR 32
	#define NP 32
	#define NX 500
	#define NY 500
	#define CZ 64
	#define CYM 64
	#define CXM 64
	#define LARGE_N 1000
	#define LENGTH 50
	#define TSTEPS 2
	#define ITER 2
	#define MAXGRID 8
#endif
#ifdef STANDARD_DATASET
	#define N 1024
	#define M 1024
	#define NI 1024
	#define NJ 1024
	#define NK 1024
	#define NL 1024
	#define NM 1024
	#define NQ 128
	#define NR 128
	#define NP 128
	#define NX 4000
	#define NY 4000
	#define CZ 256
	#define CYM 256
	#define CXM 256
	#define LARGE_N 10000
	#define LENGTH 50
	#define TSTEPS 2
	#define ITER 10
	#define MAXGRID 32
#endif
#ifdef LARGE_DATASET
	#define N 2048
	#define M 2048
	#define NI 2048
	#define NJ 2048
	#define NK 2048
	#define NL 2048
	#define NM 2048
	#define NQ 256
	#define NR 256
	#define NP 256
	#define NX 4096
	#define NY 4096
	#define CZ 512
	#define CYM 512
	#define CXM 512
	#define LARGE_N 2048*2048
	#define LENGTH 500
	#define TSTEPS 5
	#define ITER 100
	#define MAXGRID 128
#endif
#ifdef EXTRALARGE_DATASET
	#define N 4000
	#define M 4000
	#define NI 4000
	#define NJ 4000
	#define NK 4000
	#define NL 4000
	#define NM 4000
	#define NQ 1000
	#define NR 1000
	#define NP 1000
	#define NX 100000
	#define NY 100000
	#define CZ 1000
	#define CYM 1000
	#define CXM 1000
	#define LARGE_N 10000000
	#define LENGTH 500
	#define TSTEPS 10
	#define ITER 1000
	#define MAXGRID 512
#endif

