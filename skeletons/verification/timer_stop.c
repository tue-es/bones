  
  // Stop the timer for the measurement of the original code's execution time
  struct timeval bones_end_time;
  gettimeofday(&bones_end_time, NULL);
  float bones_timer = 0.001 * (1000000*(bones_end_time.tv_sec-bones_start_time.tv_sec)+bones_end_time.tv_usec-bones_start_time.tv_usec);
  printf(">>>\t\t\t Execution time [original     ]: %.3lf ms.\n", bones_timer);
