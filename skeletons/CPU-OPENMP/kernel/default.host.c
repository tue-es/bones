  
  // Run multiple OpenMP threads
  int bones_thread_count = omp_get_num_procs();
  omp_set_num_threads(bones_thread_count);
  #pragma omp parallel
  {
    int bones_thread_id = omp_get_thread_num();
    
    // Start the kernel
    bones_kernel_<algorithm_name>_0(bones_thread_id, bones_thread_count, <devicenames>, <argument_name>);
  }
