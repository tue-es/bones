  
  // Store the initial value
  cl_mem bones_initial_value = clCreateBuffer(bones_context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(<out0_type>),<out0_name>,&bones_errors); error_check(bones_errors);
  
  // Create the kernels
  cl_kernel bones_kernel_<algorithm_name>_0 = clCreateKernel(bones_program, "bones_kernel_<algorithm_name>_0", &bones_errors); error_check(bones_errors);
  cl_kernel bones_kernel_<algorithm_name>_1 = clCreateKernel(bones_program, "bones_kernel_<algorithm_name>_1", &bones_errors); error_check(bones_errors);
  cl_kernel bones_kernel_<algorithm_name>_2 = clCreateKernel(bones_program, "bones_kernel_<algorithm_name>_2", &bones_errors); error_check(bones_errors);
  
  // Run either one kernel or multiple kernels
  if (<in0_dimensions> <= 512) {
    
    // Set all the arguments to the kernel function
    int bones_num_args = 3;
    int bones_dimensions = <in0_dimensions>;
    clSetKernelArg(bones_kernel_<algorithm_name>_0,0,sizeof(bones_dimensions),(void*)&bones_dimensions);
    clSetKernelArg(bones_kernel_<algorithm_name>_0,1,sizeof(<in0_devicename>),(void*)&<in0_devicename>);
    clSetKernelArg(bones_kernel_<algorithm_name>_0,2,sizeof(<out0_devicename>),(void*)&<out0_devicename>);
    <kernel_argument_list_constants>
    // Start only one kernel
    const int bones_num_threads = DIV_CEIL(<in0_dimensions>,2);
    size_t bones_local_worksize1[] = {bones_num_threads};
    size_t bones_global_worksize1[] = {bones_num_threads};
    bones_errors = clEnqueueNDRangeKernel(bones_queue,bones_kernel_<algorithm_name>_0,1,NULL,bones_global_worksize1,bones_local_worksize1,0,NULL,&bones_event); error_check(bones_errors);
    
  }
  else {
    
    // Allocate space for an intermediate array
    cl_mem bones_device_temp = clCreateBuffer(bones_context,CL_MEM_READ_WRITE,128*sizeof(<out0_type>),NULL,&bones_errors); error_check(bones_errors);
    
    // Set all the arguments to the kernel function
    int bones_num_args = 3;
    int bones_dimensions = <in0_dimensions>;
    clSetKernelArg(bones_kernel_<algorithm_name>_0,0,sizeof(bones_dimensions),(void*)&bones_dimensions);
    clSetKernelArg(bones_kernel_<algorithm_name>_0,1,sizeof(<in0_devicename>),(void*)&<in0_devicename>);
    clSetKernelArg(bones_kernel_<algorithm_name>_0,2,sizeof(bones_device_temp),(void*)&bones_device_temp);
    <kernel_argument_list_constants>
    // Start the first kernel
    size_t bones_local_worksize1[] = {256};
    size_t bones_global_worksize1[] = {256*128};
    bones_errors = clEnqueueNDRangeKernel(bones_queue,bones_kernel_<algorithm_name>_0,1,NULL,bones_global_worksize1,bones_local_worksize1,0,NULL,&bones_event); error_check(bones_errors);
    
    // Set all the arguments to the kernel function
    clSetKernelArg(bones_kernel_<algorithm_name>_1,0,sizeof(bones_device_temp),(void*)&bones_device_temp);
    clSetKernelArg(bones_kernel_<algorithm_name>_1,1,sizeof(<out0_devicename>),(void*)&<out0_devicename>);
    // Start the second kernel
    size_t bones_local_worksize2[] = {128};
    size_t bones_global_worksize2[] = {128};
    bones_errors = clEnqueueNDRangeKernel(bones_queue,bones_kernel_<algorithm_name>_1,1,NULL,bones_global_worksize2,bones_local_worksize2,0,NULL,&bones_event); error_check(bones_errors);
    clReleaseMemObject(bones_device_temp);
  }
  
  // Set all the arguments to the kernel function
  clSetKernelArg(bones_kernel_<algorithm_name>_2,0,sizeof(bones_initial_value),(void*)&bones_initial_value);
  clSetKernelArg(bones_kernel_<algorithm_name>_2,1,sizeof(<out0_devicename>),(void*)&<out0_devicename>);
  // Perform the last computation (only needed if there is an initial value)
  size_t bones_local_worksize3[] = {1};
  size_t bones_global_worksize3[] = {1};
  bones_errors = clEnqueueNDRangeKernel(bones_queue,bones_kernel_<algorithm_name>_2,1,NULL,bones_global_worksize3,bones_local_worksize3,0,NULL,&bones_event); error_check(bones_errors);
  clReleaseMemObject(bones_initial_value);
  
  // Synchronize and clean-up the kernels
  clFinish(bones_queue);
  clReleaseKernel(bones_kernel_<algorithm_name>_0);
  clReleaseKernel(bones_kernel_<algorithm_name>_1);
  clReleaseKernel(bones_kernel_<algorithm_name>_2);
