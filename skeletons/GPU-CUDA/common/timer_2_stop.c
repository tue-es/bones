  
  // Stop the timer for the measurement of the kernel execution time
  cudaThreadSynchronize();
  cudaEvent_t bones_stop2;
  cudaEventCreate(&bones_stop2);
  cudaEventRecord(bones_stop2,0);
  cudaEventSynchronize(bones_stop2);
  float bones_timer2 = 0;
  cudaEventElapsedTime(&bones_timer2,bones_start2,bones_stop2);
  printf(">>>\t\t (<algorithm_basename>): Execution time [kernel       ]: %.3lf ms \n", bones_timer2);
