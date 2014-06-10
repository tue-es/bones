/* STARTDEF
void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>);
ENDDEF */

template<int SCALE>
__global__ void bones_kernel_<algorithm_name>_0(int *<in0_name>_index, <in0_type> *<in0_name>_value, int *<out0_name>, int const votecount)
{
	int nbins = <out0_dimension0_sum>;
	int nbins_part = ceilf((float)nbins / gridDim.y);
	int part_offset = blockIdx.y * nbins_part;
	
	//init temp. vote line in shared memory
	extern __shared__ int votespace_line[];
	for(int i=threadIdx.x; i<nbins_part*SCALE; i+=1024)
		votespace_line[i] = 0;
	__syncthreads();
	
	// calculate start and stop index of input for sub-vote spaces
	int start_index =       blockIdx.z   *votecount/gridDim.z + threadIdx.x;
	int stop_index =  min( (blockIdx.z+1)*votecount/gridDim.z , votecount);
	
	for(int i=start_index; i<stop_index; i+=1024)
	{
		//int arr_val_index = <in0_name>_index[i];
		<in0_type> arr_val_value = <in0_name>_value[i];
		int vote_index = (int)((arr_val_value & 0x00FF) * (nbins / 256.0f));
		vote_index = SCALE*vote_index + (threadIdx.x & (SCALE-1)) - part_offset;
		int vote_value = 1; // Vote value
		if(vote_index<(nbins_part*SCALE) && vote_index>=0)
			atomicAdd(&votespace_line[vote_index], vote_value);
	}
	__syncthreads();
	
	for(int i=threadIdx.x; i<nbins_part; i+=1024)
	{
		int value=0;
		#pragma unroll
		for(int j=0; j<SCALE; j++)
			value += votespace_line[SCALE*i+j];
		
		<out0_name>[blockIdx.z*nbins*gridDim.x +
				  blockIdx.x*nbins           + 
		          blockIdx.y*nbins_part      + i] = value;
	}
}

__global__ void bones_kernel_<algorithm_name>_1(int *in, int *out, int const num_subvotespaces, int const nbins)
{
	// Identify the thread
	int p = blockIdx.x*blockDim.x + threadIdx.x;
	if(p>nbins)
		return;

	// Sum the sub-votespaces
	int result = 0;
	#pragma unroll	
	for (int i=0;i<num_subvotespaces;i++) {
		result += in[blockIdx.y*num_subvotespaces*nbins + i*nbins + p];
	}
	
	// Write the results to off-chip memory
	out[blockIdx.y*nbins + p] = result;
}

// Function to start the kernel
extern "C" void bones_prekernel_<algorithm_name>_0(<devicedefinitions>, <argument_definition>) {
	int * gpu_array_index = 0;
	<in0_type> *gpu_array_value = <in0_name>;
	int cpu_votecount = <in0_dimensions>;
	int *gpu_votespace = (int*)<out0_name>;
	int *gpu_temp = 0;
		
	int nbins = <out0_dimension0_sum>;
	int number_multiprocessors = 14;
	int nbingroups = 1;

	int scaling=8192/nbins;
	int split_in_parts = 1;
	int subvotespaces = 1;
	int *gpu_out;
		
	//calculate the scaling factor, and limit it to the values 1, 2, 4, 8, 16, 32, 64 and 128
	if(scaling < 1)	{
		//too many bins requested, no scaling but splitting
		scaling = 1;
		split_in_parts = ceil(nbins / 8192.0f);
	}
	else if (scaling > 256)	{
		scaling = 256;
	}
	else	{
		int mask = 8192;
		while(0 == (mask & scaling))
			mask >>= 1;
		scaling = mask;
	}
	
	if( (nbingroups*split_in_parts) < number_multiprocessors)	{
		int const maxsub = ceil((float)(<in0_dimensions>) / (float)(32*250));
		cudaMalloc((void**)&gpu_temp, maxsub*nbingroups*nbins*sizeof(int));
		if (gpu_temp != NULL) {
			subvotespaces = number_multiprocessors / (nbingroups*split_in_parts);
			gpu_out = gpu_temp;
		}
		else {
			gpu_out = gpu_votespace;
		}
	}
	else
	{
		gpu_out = gpu_votespace;
	}
	
	//scaling = 256;
	//printf("%d %d %d %d %d\n", nbins, scaling, nbingroups, split_in_parts, subvotespaces);
	
	dim3 dimensionsBlock1(1024);
	dim3 dimensionsGrid1(nbingroups, split_in_parts, subvotespaces);
	int const nbins_part = ceilf((float)nbins / split_in_parts);
	
	switch(scaling) {
		case 256:
			bones_kernel_<algorithm_name>_0<256><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		case 128:
			bones_kernel_<algorithm_name>_0<128><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		case 64:
			bones_kernel_<algorithm_name>_0< 64><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		case 32:
			bones_kernel_<algorithm_name>_0< 32><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		case 16:
			bones_kernel_<algorithm_name>_0< 16><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		case 8:
			bones_kernel_<algorithm_name>_0<  8><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		case 4:
			bones_kernel_<algorithm_name>_0<  4><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		case 2:
			bones_kernel_<algorithm_name>_0<  2><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
		default:
			bones_kernel_<algorithm_name>_0<  1><<<dimensionsGrid1, dimensionsBlock1, scaling*nbins_part*sizeof(int)>>>
			(gpu_array_index, gpu_array_value, gpu_out, cpu_votecount);
			break;
	}
		
	if(subvotespaces > 1)	{
		dim3 dimensionsBlock2(min(nbins,1024));
		dim3 dimensionsGrid2(ceil((float)nbins/(float)1024), nbingroups);
		bones_kernel_<algorithm_name>_1<<<dimensionsGrid2, dimensionsBlock2>>>(gpu_out, gpu_votespace, subvotespaces, nbins);
		cudaFree(gpu_temp);
	}
}
