    
    // Stop the timer for the measurement of the kernel execution time
    gettimeofday(&bones_end_time2, NULL);
    bones_timer2 += 0.001 * (1000000*(bones_end_time2.tv_sec-bones_start_time2.tv_sec)+bones_end_time2.tv_usec-bones_start_time2.tv_usec);
  }
  
  // Print the measurement data
  printf(">>>\t\t (<algorithm_basename>): Execution time [kernel       ]: %.3lf ms \n", bones_timer2/((float)ITERS));
