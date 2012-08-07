  
  // Start the timer for the measurement of the kernel execution time
  cudaThreadSynchronize();
  cudaEvent_t bones_start2;
  cudaEventCreate(&bones_start2);
  cudaEventRecord(bones_start2,0);
