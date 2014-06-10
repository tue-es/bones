
// Start of the <algorithm_name> kernel
__kernel void bones_kernel_<algorithm_name>_0(<devicedefinitionsopencl>, <argument_definition>) {
  const int bones_global_id = get_global_id(0);
  if (bones_global_id < (<parallelism>)) {
    
    // Calculate the global ID(s) based on the thread id
    <ids>
    
    // Start the computation
<algorithm_code1>
  }
}
