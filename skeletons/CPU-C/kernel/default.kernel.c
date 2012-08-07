/* STARTDEF
void bones_kernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>);
ENDDEF */
// Start of the <algorithm_name> kernel
void bones_kernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
  for(int bones_global_id=0; bones_global_id<<parallelism>; bones_global_id++) {
    
    // Calculate the global ID(s) based on the thread id
    <ids>
    
    // Perform the main computation
<algorithm_code1>
  }
}

