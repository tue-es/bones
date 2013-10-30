
////////////////////////////////////////
//////////// Globals ///////////////////
////////////////////////////////////////

#define BONES_MIN(a,b) ((a<b) ? a : b)
#define BONES_MAX(a,b) ((a>b) ? a : b)
#define DIV_CEIL(a,b)  ((a+b-1)/b)
#define DIV_FLOOR(a,b) (a/b)

// CUDA timers
cudaEvent_t bones_start2;
cudaEvent_t bones_stop2;

// Function to initialize the GPU (for fair measurements, streams, timers)
void bones_initialize_target(void) {
  int* bones_temporary = 0;
  cudaMalloc((void**)&bones_temporary, sizeof(int));
  cudaFree(bones_temporary);
  cudaStreamCreate(&kernel_stream);
  cudaEventCreate(&bones_start2);
  cudaEventCreate(&bones_stop2);
}

// Declaration of the original function
int bones_main(void);

////////////////////////////////////////
//////////// Main function /////////////
////////////////////////////////////////

// New main function for initialisation and clean-up
int main(void) {
  
  // Initialisation of the scheduler
  bones_initialize_scheduler();
  pthread_t bones_scheduler_thread;
  pthread_create(&bones_scheduler_thread, NULL, bones_scheduler, NULL);
  
  // Initialisation of the target
  bones_initialize_target();
  
  // Original main function
  int bones_return = bones_main();
  
  // Clean-up
  bones_scheduler_done = 1;
  pthread_join(bones_scheduler_thread, NULL);
  cudaStreamDestroy(kernel_stream);
  return bones_return;
}

////////////////////////////////////////
////////// Accelerated functions ///////
////////////////////////////////////////
