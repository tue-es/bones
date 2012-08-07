/* STARTDEF
void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>);
ENDDEF */
// Start of the <algorithm_name> kernel (main, not unrolled kernel)
__global__ void bones_kernel_<algorithm_name>_0(int bones_input_size, <in0_type><in0_devicepointer> <in0_name>, <out0_type><out0_devicepointer> <out0_name>, <argument_definition>) {
  const int bones_threadblock_work = DIV_CEIL(bones_input_size,gridDim.x);
  const int bones_parallel_work = BONES_MIN(blockDim.x,bones_threadblock_work);
  const int bones_sequential_work = DIV_CEIL(bones_threadblock_work,bones_parallel_work);
  const int bones_local_id = threadIdx.x;
  const int bones_global_id = blockIdx.x*bones_parallel_work + threadIdx.x;
  <ids>
  int bones_iter_id = <in0_flatindex>;
  
  // Load data into thread private memory and perform the first computation(s) sequentially
  <in0_type> bones_temporary = <in0_name>[bones_iter_id];
  <in0_type> bones_private_memory = <algorithm_code3>;
  for(int c=1; c<bones_sequential_work; c++) {
    bones_iter_id = bones_iter_id + bones_parallel_work*gridDim.x<factors>;
    if (bones_iter_id <= <in0_to>) {
      bones_temporary = <in0_name>[bones_iter_id];
      bones_private_memory = <algorithm_code1>;
    }
  }
  
  // Initialize the local memory
  volatile __shared__ <in0_type> bones_local_memory[512];
  bones_local_memory[bones_local_id] = bones_private_memory;
  __syncthreads();
  
  // Perform the remainder of the computations in parallel using a parallel reduction tree
  int bones_offset_id;
  for (int c=512; c>=2; c=c>>1) {
    if ((2*bones_parallel_work > c) && (threadIdx.x < c/2)) {
      bones_offset_id = threadIdx.x+c/2;
      if (bones_offset_id < bones_parallel_work) {
        __syncthreads();
        bones_local_memory[bones_local_id] = <algorithm_code2>;
      }
    }
    __syncthreads();
  }
  
  // Write the final result back to the global memory
  if (threadIdx.x == 0) { <out0_name>[blockIdx.x] = bones_local_memory[0]; }
}

// Start of the <algorithm_name> kernel (secondary, not unrolled kernel)
__global__ void bones_kernel_<algorithm_name>_1(<in0_type><in0_devicepointer> <in0_name>, <out0_type><out0_devicepointer> <out0_name>, <argument_definition>) {
  const int bones_local_id = threadIdx.x;
  const int bones_global_id = threadIdx.x;
  
  // Initialize the local memory
  volatile __shared__ <in0_type> bones_local_memory[512];
  bones_local_memory[bones_local_id] = <in0_name>[bones_global_id];
  __syncthreads();
  
  // Perform reduction using a parallel reduction tree
  int bones_offset_id;
  for (int c=128; c>=2; c=c>>1) {
    if (threadIdx.x < c/2) {
      bones_offset_id = threadIdx.x+c/2;
      bones_local_memory[bones_local_id] = <algorithm_code2>;
      __syncthreads();
    }
  }
  
  // Write the final result back to the global memory
  if (threadIdx.x == 0) { <out0_name>[0] = bones_local_memory[0]; }
}

// Start of the <algorithm_name> kernel (final, initial value kernel)
__global__ void bones_kernel_<algorithm_name>_2(<out0_type><out0_devicepointer> bones_initial_value, <out0_type><out0_devicepointer> <out0_name>, <argument_definition>) {
  <out0_type> bones_private_memory = <out0_name>[0];
  <out0_type> bones_temporary = bones_initial_value[0];
  <out0_name>[0] = <algorithm_code4>;
}

// Function to start the kernel
extern "C" void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
  
  // Store the initial value
  <out0_type>* bones_initial_value = 0;
  cudaMalloc(&bones_initial_value, sizeof(<out0_type>));
  cudaMemcpy(bones_initial_value, <out0_name>, sizeof(<out0_type>), cudaMemcpyDeviceToDevice);
  
  // Run either one kernel or multiple kernels
  if (<in0_dimensions> <= 1024) {
    
    // Start only one kernel
    const int bones_num_threads = DIV_CEIL(<in0_dimensions>,2);
    dim3 bones_threads(bones_num_threads);
    dim3 bones_grid(1);
    bones_kernel_<algorithm_name>_0<<< bones_grid, bones_threads >>>(<in0_dimensions>,<in0_name>,<out0_name>,<argument_name>);
  }
  else {
    
    // Allocate space for an intermediate array
    <out0_type>* bones_device_temp = 0;
    cudaMalloc(&bones_device_temp, 128*sizeof(<out0_type>));
    
    // Start the first kernel
    dim3 bones_threads1(512);
    dim3 bones_grid1(128);
    bones_kernel_<algorithm_name>_0<<< bones_grid1, bones_threads1 >>>(<in0_dimensions>,<in0_name>,bones_device_temp,<argument_name>);
    
    // Start the second kernel
    dim3 bones_threads2(128);
    dim3 bones_grid2(1);
    bones_kernel_<algorithm_name>_1<<< bones_grid2, bones_threads2 >>>(bones_device_temp,<out0_name>,<argument_name>);
    
    cudaFree(bones_device_temp);
  }
  
  // Perform the last computation (only needed if there is an initial value)
  dim3 bones_threads3(1);
  dim3 bones_grid3(1);
  bones_kernel_<algorithm_name>_2<<< bones_grid3, bones_threads3 >>>(bones_initial_value,<out0_name>,<argument_name>);
  cudaFree(bones_initial_value);
}
