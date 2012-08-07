#include <stdio.h>
#include <cuda_runtime.h>

#define BONES_MIN(a,b) ((a<b) ? a : b)
#define BONES_MAX(a,b) ((a>b) ? a : b)
#define DIV_CEIL(a,b)  ((a+b-1)/b)
#define DIV_FLOOR(a,b) (a/b)

// Function to initialize the GPU (for fair measurements)
void bones_initialize_target(void) {
  int* bones_temporary = 0;
  cudaMalloc((void**)&bones_temporary, sizeof(int));
  cudaFree(bones_temporary);
}

// Declaration of the original function
int bones_main(void);

// New main function for initialisation and clean-up
int main(void) {
  
  // Initialisation of the target
  bones_initialize_target();
  
  // Original main function
  int bones_return = bones_main();
  
  // Clean-up
  return bones_return;
}

