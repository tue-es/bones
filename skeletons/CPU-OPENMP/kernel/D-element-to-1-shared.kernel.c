/* STARTDEF
void bones_kernel_<algorithm_name>_0(int bones_thread_id, int bones_thread_count, int bones_size, <devicedefinitions>, <argument_definition>);
void bones_kernel_<algorithm_name>_1(int bones_size, <devicedefinitions>, <argument_definition>);
void bones_kernel_<algorithm_name>_2(<out0_type> bones_initial_value, <out0_type><out0_devicepointer> <out0_name>, <argument_definition>);
ENDDEF */
// Start of the <algorithm_name> kernel (main part)
void bones_kernel_<algorithm_name>_0(int bones_thread_id, int bones_thread_count, int bones_size, <devicedefinitions>, <argument_definition>) {
  const int bones_work = DIV_CEIL(bones_size,bones_thread_count);
  const int bones_global_id = bones_thread_id;
  <ids>
  int bones_iter_id = <in0_flatindex>;
  
  // Use a thread private memory to perform the per-thread computation(s)
  <in0_type> bones_temporary = <in0_name>[bones_iter_id];
  <in0_type> bones_private_memory = <algorithm_code2>;
  for(int c=1; c<bones_work; c++) {
    bones_iter_id = bones_iter_id + bones_thread_count<factors>;
    if (bones_iter_id <= <in0_to>) {
      bones_temporary = <in0_name>[bones_iter_id];
      bones_private_memory = <algorithm_code1>;
    }
  }
  
  // Store the result
  <out0_name>[bones_thread_id] = bones_private_memory;
}

// Start of the <algorithm_name> kernel (secondary part)
void bones_kernel_<algorithm_name>_1(int bones_size, <devicedefinitions>, <argument_definition>) {
  
  // Use a private memory to perform the sequential computation(s)
  <in0_type> bones_private_memory = <in0_name>[0];
  for(int bones_iter_id=1; bones_iter_id<bones_size; bones_iter_id++) {
    bones_private_memory = bones_private_memory + <in0_name>[bones_iter_id];
  }
  
  // Store the result
  <out0_name>[0] = bones_private_memory;
}

// Start of the <algorithm_name> kernel (final, initial value kernel)
void bones_kernel_<algorithm_name>_2(<out0_type> bones_initial_value, <out0_type><out0_devicepointer> <out0_name>, <argument_definition>) {
  <out0_type> bones_private_memory = <out0_name>[0];
  <out0_type> bones_temporary = bones_initial_value;
  <out0_name>[0] = <algorithm_code3>;
}
