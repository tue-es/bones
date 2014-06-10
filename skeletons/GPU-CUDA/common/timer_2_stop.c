  
  // Stop the timer for the measurement of the kernel execution time
  //cudaStreamSynchronize(kernel_stream);
  cudaEventRecord(bones_stop2,kernel_stream);
  cudaEventSynchronize(bones_stop2);
  float bones_timer2 = 0;
  cudaEventElapsedTime(&bones_timer2,bones_start2,bones_stop2);
  printf(">>>\t\t Execution time [kernel <algorithm_basename>]: %.3lf ms \n", bones_timer2);
