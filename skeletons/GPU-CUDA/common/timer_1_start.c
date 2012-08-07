  
  // Start the timer for the measurement of the kernel and memory copy execution time
  cudaThreadSynchronize();
  cudaEvent_t bones_start1;
  cudaEventCreate(&bones_start1);
  cudaEventRecord(bones_start1,0);
