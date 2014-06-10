/* STARTDEF
void bones_kernel_<algorithm_name>_0(int bones_thread_id, int bones_thread_count, <devicedefinitions>, <argument_definition>);
ENDDEF */
// Start of the <algorithm_name> kernel
void bones_kernel_<algorithm_name>_0(int bones_thread_id, int bones_thread_count, <devicedefinitions>, <argument_definition>) {
  int bones_workload = DIV_CEIL(<parallelism>,bones_thread_count);
  int bones_start = bones_thread_id*bones_workload;
  int bones_end = BONES_MIN((bones_thread_id+1)*bones_workload,<parallelism>);
  for(int bones_global_id=bones_start; bones_global_id<bones_end; bones_global_id++) {
    
    // Calculate the global ID(s) based on the thread id
    <ids>
    
    // Perform the main computation
<algorithm_code1>
  }
}

