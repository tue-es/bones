  
  // Flush the CPU cache (for measurement purposes only)
  const int bones_flush_size = 4*1024*1024; // (16MB)
  int bones_flush_i;
  int bones_flush_j;
  char *bones_flush_c = (char *)malloc(bones_flush_size);
  for (bones_flush_i=0; bones_flush_i<10; bones_flush_i++) {
    for (bones_flush_j=0; bones_flush_j<bones_flush_size; bones_flush_j++) {
      bones_flush_c[bones_flush_j] = bones_flush_i*bones_flush_j;
    }
  }
  free(bones_flush_c);
