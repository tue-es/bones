  
  // Start the timer for the measurement of the kernel execution time
  clFinish(bones_queue);
  for (int bones_iter=0; bones_iter<ITERS; bones_iter++) {