#include <omp.h>
#include <stdlib.h>

#define BONES_MIN(a,b) ((a<b) ? a : b)
#define BONES_MAX(a,b) ((a>b) ? a : b)
#define DIV_CEIL(a,b)  ((a+b-1)/b)
#define DIV_FLOOR(a,b) (a/b)

// Multiple iterations for kernel measurements
#define ITERS 1

// Function to initialize the CPU platform (for fair measurements)
void bones_initialize_target(void) {
  int bones_thread_count = omp_get_num_procs();
  omp_set_num_threads(bones_thread_count);
  #pragma omp parallel
  {
    int bones_thread_id = omp_get_thread_num();
  }
}

// Declaration of the original function
int bones_main(void);

// New main function for initialisation and clean-up
int main(void) {
  
  // Initialisation
  bones_initialize_target();
  
  // Original main function
  int bones_return = bones_main();
  
  // Clean-up
  return bones_return;
}

