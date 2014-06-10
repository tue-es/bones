  
  // End the timer for the measurement of the kernel and memory copy execution time
  #if (ITERS == 1)
    clFinish(bones_queue);
    struct timeval bones_end_time1;
    gettimeofday(&bones_end_time1, NULL);
    float bones_timer1 = 0.001 * (1000000*(bones_end_time1.tv_sec-bones_start_time1.tv_sec)+bones_end_time1.tv_usec-bones_start_time1.tv_usec);
    printf(">>>\t\t (<algorithm_basename>): Execution time [kernel+memcpy]: %.3lf ms \n", bones_timer1);
  #endif