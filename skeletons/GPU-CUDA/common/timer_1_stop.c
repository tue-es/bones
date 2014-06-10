  
  // End the timer for the measurement of the kernel and memory copy execution time
  cudaThreadSynchronize();
  cudaEvent_t bones_stop1;
  cudaEventCreate(&bones_stop1);
  cudaEventRecord(bones_stop1,0);
  cudaEventSynchronize(bones_stop1);
  float bones_timer1 = 0;
  cudaEventElapsedTime(&bones_timer1,bones_start1,bones_stop1);
  printf(">>>\t\t (<algorithm_basename>): Execution time [kernel+memcpy]: %.3lf ms \n", bones_timer1);
