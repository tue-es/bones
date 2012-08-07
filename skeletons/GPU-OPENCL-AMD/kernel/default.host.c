  
  // Create the kernel
  cl_kernel bones_kernel_<algorithm_name>_0 = clCreateKernel(bones_program, "bones_kernel_<algorithm_name>_0", &bones_errors); error_check(bones_errors);
  
  // Set all the arguments to the kernel function
  int bones_num_args = 0;
  <kernel_argument_list>
  // Start the kernel
  size_t bones_global_worksize[] = {<parallelism>};
  bones_errors = clEnqueueNDRangeKernel(bones_queue,bones_kernel_<algorithm_name>_0,1,NULL,bones_global_worksize,NULL,0,NULL,&bones_event); error_check(bones_errors);
  
  // Synchronize and clean-up the kernel
  clFinish(bones_queue);
  clReleaseKernel(bones_kernel_<algorithm_name>_0);
