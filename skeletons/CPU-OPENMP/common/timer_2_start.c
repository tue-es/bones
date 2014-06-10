  // Initialize the timer
  float bones_timer2 = 0;
  struct timeval bones_start_time2;
  struct timeval bones_end_time2;
  for (int bones_iter=0; bones_iter<ITERS; bones_iter++) {
    
    // Flush the CPU cache (for measurement purposes only)
    const int bones_flush_size = 4*1024*1024; // (16MB)
    char *bones_flush_c = (char *)malloc(bones_flush_size);
    for (int i=0; i<10; i++) {
      for (int j=0; j<bones_flush_size; j++) {
        bones_flush_c[j] = i*j;
      }
    }
    free(bones_flush_c);
    
    // Start the timer for the measurement of the kernel execution time
    gettimeofday(&bones_start_time2, NULL);