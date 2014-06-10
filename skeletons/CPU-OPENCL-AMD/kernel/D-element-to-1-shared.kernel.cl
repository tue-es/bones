
// Start of the <algorithm_name> kernel (main, not unrolled kernel)
__kernel void bones_kernel_<algorithm_name>_0(int bones_input_size, __global <in0_type><in0_devicepointer> <in0_name>, __global <out0_type><out0_devicepointer> <out0_name>, <argument_definition>) {
  const int bones_threadblock_work = DIV_CEIL(bones_input_size,get_num_groups(0));
  const int bones_parallel_work = BONES_MIN(get_local_size(0),bones_threadblock_work);
  const int bones_sequential_work = DIV_CEIL(bones_threadblock_work,bones_parallel_work);
  const int bones_local_id = get_local_id(0);
  const int bones_global_id = get_global_id(0);
  <ids>
  int bones_iter_id = <in0_flatindex>;
  
  // Load data into thread private memory and perform the first computation(s) sequentially
  <in0_type> bones_temporary = <in0_name>[bones_iter_id];
  <in0_type> bones_private_memory = <algorithm_code3>;
  for(int c=1; c<bones_sequential_work; c++) {
    bones_iter_id = bones_iter_id + bones_parallel_work*get_num_groups(0)<factors>;
    if (bones_iter_id <= <in0_to>) {
      bones_temporary = <in0_name>[bones_iter_id];
      bones_private_memory = <algorithm_code1>;
    }
  }
  // Initialize the local memory
  volatile __local <in0_type> bones_local_memory[256];
  bones_local_memory[bones_local_id] = bones_private_memory;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // Perform the remainder of the computations in parallel using a parallel reduction tree
  int bones_offset_id;
  for (int c=256; c>=2; c=c>>1) {
    if ((2*bones_parallel_work > c) && (get_local_id(0) < c/2)) {
      bones_offset_id = get_local_id(0)+c/2;
      if (bones_offset_id < bones_parallel_work) {
        bones_local_memory[bones_local_id] = <algorithm_code2>;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  // Write the final result back to the global memory
  if (get_local_id(0) == 0) { <out0_name>[get_group_id(0)] = bones_local_memory[0]; }
}

// Start of the <algorithm_name> kernel (secondary, not unrolled kernel)
__kernel void bones_kernel_<algorithm_name>_1(__global <in0_type><in0_devicepointer> <in0_name>, __global <out0_type><out0_devicepointer> <out0_name>) {
  const int bones_local_id = get_local_id(0);
  const int bones_global_id = get_local_id(0);
  
  // Initialize the local memory
  volatile __local <in0_type> bones_local_memory[128];
  bones_local_memory[bones_local_id] = <in0_name>[bones_global_id];
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // Perform reduction using a parallel reduction tree
  int bones_offset_id;
  for (int c=128; c>=2; c=c>>1) {
    if (get_local_id(0) < c/2) {
      bones_offset_id = get_local_id(0)+c/2;
      bones_local_memory[bones_local_id] = <algorithm_code2>;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  // Write the final result back to the global memory
  if (get_local_id(0) == 0) { <out0_name>[0] = bones_local_memory[0]; }
}

// Start of the <algorithm_name> kernel (final, initial value kernel)
__kernel void bones_kernel_<algorithm_name>_2(__global <out0_type><out0_devicepointer> bones_initial_value, __global <out0_type><out0_devicepointer> <out0_name>) {
  <out0_type> bones_private_memory = <out0_name>[0];
  <out0_type> bones_temporary = bones_initial_value[0];
  <out0_name>[0] = <algorithm_code4>;
}
