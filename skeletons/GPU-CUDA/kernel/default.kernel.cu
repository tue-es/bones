/* STARTDEF
void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>);
ENDDEF */
// Start of the <algorithm_name> kernel
__global__ void bones_kernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
  const int bones_global_id = blockIdx.x*blockDim.x + threadIdx.x;
  if (bones_global_id < (<parallelism>)) {
    
    // Calculate the global ID(s) based on the thread id
    <ids>
    
    // Start the computation
<algorithm_code1>
  }
}

// Function to start the kernel
extern "C" void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
  int bones_block_size;
  if      (<parallelism> >= 64*512) { bones_block_size = 512;}
  else if (<parallelism> >= 64*256) { bones_block_size = 256;}
  else if (<parallelism> >= 64*128) { bones_block_size = 128;}
  else if (<parallelism> >= 64*64 ) { bones_block_size = 64; }
  else { bones_block_size = 32; }
  dim3 bones_threads(bones_block_size);
  dim3 bones_grid(DIV_CEIL(<parallelism>,bones_block_size));
  bones_kernel_<algorithm_name>_0<<< bones_grid, bones_threads >>>(<names>, <argument_name>);
}
