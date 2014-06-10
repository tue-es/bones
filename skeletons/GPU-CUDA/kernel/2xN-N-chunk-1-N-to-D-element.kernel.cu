/* STARTDEF
void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>);
ENDDEF */
#define SHUFFLE_X 16
#define SHUFFLE_Y 16

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

// Start of the <algorithm_name> kernel (pre-kernel for shuffling) - for first input
__global__ void bones_kernel_<algorithm_name>_1(<in0_type><in0_devicepointer> <in0_name>, <in0_type><in0_devicepointer> shuffled_<in0_name>, <argument_definition>) {
  const int bones_global_id_0 = blockIdx.x*blockDim.x + threadIdx.x;
  const int bones_global_id_1 = blockIdx.y*blockDim.y + threadIdx.y;
  
  // Set-up the local memory for shuffling
  __shared__ <in0_type> buffer[SHUFFLE_X][SHUFFLE_Y];
  
  // Swap the x and y coordinates to perform the rotation (coalesced)
  if (bones_global_id_0 < ((<in0_dimensions>)/(<in0_parameters>)) && bones_global_id_1 < (<in0_parameters>)) {
    buffer[threadIdx.y][threadIdx.x] = <in0_name>[bones_global_id_0 + bones_global_id_1 * ((<in0_dimensions>)/(<in0_parameters>))];
  }
  
  // Synchronize all threads in the threadblock
  __syncthreads();
  
  // We don't have to swap the x and y thread indices here, because that's already done in the local memory
  const int bones_global_id_0_new = blockIdx.y*blockDim.y + threadIdx.x;
  const int bones_global_id_1_new = blockIdx.x*blockDim.x + threadIdx.y;
  
  // Store the shuffled result (coalesced)
  if (bones_global_id_0_new < ((<in0_dimensions>)/(<in0_parameters>)) && bones_global_id_1_new < (<in0_parameters>)) {
    shuffled_<in0_name>[bones_global_id_0_new + bones_global_id_1_new * <in0_parameters>] =  buffer[threadIdx.x][threadIdx.y];
  }
}

// Start of the <algorithm_name> kernel (pre-kernel for shuffling) - for second input
__global__ void bones_kernel_<algorithm_name>_2(<in1_type><in1_devicepointer> <in1_name>, <in1_type><in1_devicepointer> shuffled_<in1_name>, <argument_definition>) {
  const int bones_global_id_0 = blockIdx.x*blockDim.x + threadIdx.x;
  const int bones_global_id_1 = blockIdx.y*blockDim.y + threadIdx.y;
  
  // Set-up the local memory for shuffling
  __shared__ <in1_type> buffer[SHUFFLE_X][SHUFFLE_Y];
  
  // Swap the x and y coordinates to perform the rotation (coalesced)
  if (bones_global_id_0 < ((<in1_dimensions>)/(<in1_parameters>)) && bones_global_id_1 < (<in1_parameters>)) {
    buffer[threadIdx.y][threadIdx.x] = <in1_name>[bones_global_id_0 + bones_global_id_1 * ((<in1_dimensions>)/(<in1_parameters>))];
  }
  
  // Synchronize all threads in the threadblock
  __syncthreads();
  
  // We don't have to swap the x and y thread indices here, because that's already done in the local memory
  const int bones_global_id_0_new = blockIdx.y*blockDim.y + threadIdx.x;
  const int bones_global_id_1_new = blockIdx.x*blockDim.x + threadIdx.y;
  
  // Store the shuffled result (coalesced)
  if (bones_global_id_0_new < ((<in1_dimensions>)/(<in1_parameters>)) && bones_global_id_1_new < (<in1_parameters>)) {
    shuffled_<in1_name>[bones_global_id_0_new + bones_global_id_1_new * <in1_parameters>] =  buffer[threadIdx.x][threadIdx.y];
  }
}

// Function to start the kernel
extern "C" void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
  int bones_block_size;
  if      (<parallelism> >= 64*512 ) { bones_block_size = 512; }
  else if (<parallelism> >= 64*256 ) { bones_block_size = 256; }
  else                               { bones_block_size = 128; }
  
  // First perform some pre-shuffling (for the first input)
  <in0_type>* shuffled_<in0_name> = 0;
  cudaMalloc((void**)&shuffled_<in0_name>, <in0_dimensions>*sizeof(<in0_type>));
  dim3 bones_threads1(SHUFFLE_X,SHUFFLE_Y);
  dim3 bones_grid1(DIV_CEIL(((<in0_dimensions>)/(<in0_parameters>)),SHUFFLE_X),DIV_CEIL(<in0_parameters>,SHUFFLE_Y));
  bones_kernel_<algorithm_name>_1<<< bones_grid1, bones_threads1 >>>(<in0_name>, shuffled_<in0_name>, <argument_name>);
  <in0_type>* temp_<in0_name> = <in0_name>;
  <in0_name> = shuffled_<in0_name>;
  //cudaFree(temp_<in0_name>);
  
  // First perform some pre-shuffling (for the second input)
  <in0_type>* shuffled_<in1_name> = 0;
  cudaMalloc((void**)&shuffled_<in1_name>, <in1_dimensions>*sizeof(<in1_type>));
  dim3 bones_threads2(SHUFFLE_X,SHUFFLE_Y);
  dim3 bones_grid2(DIV_CEIL(((<in1_dimensions>)/(<in1_parameters>)),SHUFFLE_X),DIV_CEIL(<in1_parameters>,SHUFFLE_Y));
  bones_kernel_<algorithm_name>_2<<< bones_grid2, bones_threads2 >>>(<in1_name>, shuffled_<in1_name>, <argument_name>);
  <in1_type>* temp_<in1_name> = <in1_name>;
  <in1_name> = shuffled_<in1_name>;
  //cudaFree(temp_<in1_name>);
  
  // Then run the original kernel
  dim3 bones_threads0(bones_block_size);
  dim3 bones_grid0(DIV_CEIL(<parallelism>,bones_block_size));
  bones_kernel_<algorithm_name>_0<<< bones_grid0, bones_threads0 >>>(<names>, <argument_name>);
}
