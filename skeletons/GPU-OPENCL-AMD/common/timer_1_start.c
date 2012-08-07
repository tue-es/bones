  
  // Start the timer for the measurement of the kernel and memory copy execution time
  struct timeval bones_start_time1;
  clFinish(bones_queue);
  gettimeofday(&bones_start_time1, NULL);
