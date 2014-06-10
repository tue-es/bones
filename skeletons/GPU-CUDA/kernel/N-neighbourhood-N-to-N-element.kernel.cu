/* STARTDEF
void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>);
ENDDEF */
// Start of the <algorithm_name> kernel
__global__ void bones_kernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
  const int bones_global_id = blockIdx.x*blockDim.x + threadIdx.x;
  int bones_local_id = threadIdx.x;
  if (bones_global_id < <in0_dimensions>) {
    
    // Calculate the local and global ID(s) based on the thread id
    int bones_local_id_0 = bones_local_id;
    <out0_ids>
    
    // Load the input data into local memory
    __shared__ <in0_type> bones_local_memory_<in0_name>[512+<in0_parameter0_sum>];
    bones_local_id_0 = bones_local_id_0-(<in0_parameter0_from>);
    bones_local_memory_<in0_name>[bones_local_id_0] = <in0_name>[bones_global_id_0];
    
    // Load the left border into local memory
    if (threadIdx.x < -(<in0_parameter0_from>)) {
      bones_local_memory_<in0_name>[bones_local_id_0+<in0_parameter0_from>] = <in0_name>[bones_global_id_0+<in0_parameter0_from>];
    }
    
    // Load the right border into local memory
    if ((threadIdx.x >= 512-<in0_parameter0_to>) || (bones_global_id_0 >= <in0_dimensions>-<in0_parameter0_to>)) {
      bones_local_memory_<in0_name>[bones_local_id_0+<in0_parameter0_to>] = <in0_name>[bones_global_id_0+<in0_parameter0_to>];
    }
    
    // Synchronize all the threads in a threadblock
    __syncthreads();
    
    // Perform the main computation
<algorithm_code1>
  }
}

// Function to start the kernel
extern "C" void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
  dim3 bones_threads(512);
  dim3 bones_grid(DIV_CEIL(<in0_dimensions>,512));
  bones_kernel_<algorithm_name>_0<<< bones_grid, bones_threads >>>(<names>, <argument_name>);
}
