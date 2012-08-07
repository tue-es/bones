  
  if (<in0_dimensions> > 0) {
    
    // Store the initial value
    <out0_type> bones_initial_value = <out0_name>[0];
    
    // Create a temporary array to store intermediate data
    int bones_thread_count = BONES_MIN(omp_get_num_procs(),<in0_dimensions>);
    <out0_type>* bones_temporary = (<out0_type>*)malloc(bones_thread_count*sizeof(<out0_type>));
    
    // Run multiple OpenMP threads
    omp_set_num_threads(bones_thread_count);
    #pragma omp parallel
    {
      int bones_thread_id = omp_get_thread_num();
      
      // Perform the major part of the computation in parallel
      bones_kernel_<algorithm_name>_0(bones_thread_id, bones_thread_count, <in0_dimensions>, <in_devicenames>, bones_temporary, <argument_name>);
    }
    
    // Compute the second part of the algorithm with only one thread
    bones_kernel_<algorithm_name>_1(bones_thread_count, bones_temporary, <out_devicenames>, <argument_name>);
    free(bones_temporary);
    
    // Perform the last computation (only needed if there is an initial value)
    bones_kernel_<algorithm_name>_2(bones_initial_value,<out0_name>,<argument_name>);
  }
