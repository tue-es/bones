#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>

#define BONES_MIN(a,b) ((a<b) ? a : b)
#define BONES_MAX(a,b) ((a>b) ? a : b)
#define DIV_CEIL(a,b)  ((a+b-1)/b)
#define DIV_FLOOR(a,b) (a/b)

// Multiple iterations for kernel measurements
#define ITERS 1

// Load the OpenCL kernel from file
char * get_source(const char* bones_filename) {
  FILE* bones_fp = fopen(bones_filename,"r");
  fseek(bones_fp,0,SEEK_END);
  long bones_size = ftell(bones_fp);
  rewind(bones_fp);
  char *bones_source = (char *)malloc(sizeof(char)*(bones_size+1));
  int bones_temp = fread(bones_source,1,sizeof(char)*bones_size,bones_fp);
  bones_source[bones_size] = '\0';
  fclose(bones_fp);
  return bones_source;
}

// Print an error if it occurs
void error_check(cl_int bones_errors) {
  if(bones_errors != CL_SUCCESS) {
    switch (bones_errors) {
      case CL_DEVICE_NOT_FOUND:                 printf("--- Error: Device not found.\n"); break;
      case CL_DEVICE_NOT_AVAILABLE:             printf("--- Error: Device not available\n"); break;
      case CL_COMPILER_NOT_AVAILABLE:           printf("--- Error: Compiler not available\n"); break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("--- Error: Memory object allocation failure\n"); break;
      case CL_OUT_OF_RESOURCES:                 printf("--- Error: Out of resources\n"); break;
      case CL_OUT_OF_HOST_MEMORY:               printf("--- Error: Out of host memory\n"); break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("--- Error: Profiling information not available\n"); break;
      case CL_MEM_COPY_OVERLAP:                 printf("--- Error: Memory copy overlap\n"); break;
      case CL_IMAGE_FORMAT_MISMATCH:            printf("--- Error: Image format mismatch\n"); break;
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("--- Error: Image format not supported\n"); break;
      case CL_BUILD_PROGRAM_FAILURE:            printf("--- Error: Program build failure\n"); break;
      case CL_MAP_FAILURE:                      printf("--- Error: Map failure\n"); break;
      case CL_INVALID_VALUE:                    printf("--- Error: Invalid value\n"); break;
      case CL_INVALID_DEVICE_TYPE:              printf("--- Error: Invalid device type\n"); break;
      case CL_INVALID_PLATFORM:                 printf("--- Error: Invalid platform\n"); break;
      case CL_INVALID_DEVICE:                   printf("--- Error: Invalid device\n"); break;
      case CL_INVALID_CONTEXT:                  printf("--- Error: Invalid context\n"); break;
      case CL_INVALID_QUEUE_PROPERTIES:         printf("--- Error: Invalid queue properties\n"); break;
      case CL_INVALID_COMMAND_QUEUE:            printf("--- Error: Invalid command queue\n"); break;
      case CL_INVALID_HOST_PTR:                 printf("--- Error: Invalid host pointer\n"); break;
      case CL_INVALID_MEM_OBJECT:               printf("--- Error: Invalid memory object\n"); break;
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("--- Error: Invalid image format descriptor\n"); break;
      case CL_INVALID_IMAGE_SIZE:               printf("--- Error: Invalid image size\n"); break;
      case CL_INVALID_SAMPLER:                  printf("--- Error: Invalid sampler\n"); break;
      case CL_INVALID_BINARY:                   printf("--- Error: Invalid binary\n"); break;
      case CL_INVALID_BUILD_OPTIONS:            printf("--- Error: Invalid build options\n"); break;
      case CL_INVALID_PROGRAM:                  printf("--- Error: Invalid program\n"); break;
      case CL_INVALID_PROGRAM_EXECUTABLE:       printf("--- Error: Invalid program executable\n"); break;
      case CL_INVALID_KERNEL_NAME:              printf("--- Error: Invalid kernel name\n"); break;
      case CL_INVALID_KERNEL_DEFINITION:        printf("--- Error: Invalid kernel definition\n"); break;
      case CL_INVALID_KERNEL:                   printf("--- Error: Invalid kernel\n"); break;
      case CL_INVALID_ARG_INDEX:                printf("--- Error: Invalid argument index\n"); break;
      case CL_INVALID_ARG_VALUE:                printf("--- Error: Invalid argument value\n"); break;
      case CL_INVALID_ARG_SIZE:                 printf("--- Error: Invalid argument size\n"); break;
      case CL_INVALID_KERNEL_ARGS:              printf("--- Error: Invalid kernel arguments\n"); break;
      case CL_INVALID_WORK_DIMENSION:           printf("--- Error: Invalid work dimensionsension\n"); break;
      case CL_INVALID_WORK_GROUP_SIZE:          printf("--- Error: Invalid work group size\n"); break;
      case CL_INVALID_WORK_ITEM_SIZE:           printf("--- Error: Invalid work item size\n"); break;
      case CL_INVALID_GLOBAL_OFFSET:            printf("--- Error: Invalid global offset\n"); break;
      case CL_INVALID_EVENT_WAIT_LIST:          printf("--- Error: Invalid event wait list\n"); break;
      case CL_INVALID_EVENT:                    printf("--- Error: Invalid event\n"); break;
      case CL_INVALID_OPERATION:                printf("--- Error: Invalid operation\n"); break;
      case CL_INVALID_GL_OBJECT:                printf("--- Error: Invalid OpenGL object\n"); break;
      case CL_INVALID_BUFFER_SIZE:              printf("--- Error: Invalid buffer size\n"); break;
      case CL_INVALID_MIP_LEVEL:                printf("--- Error: Invalid mip-map level\n"); break;
      default:                                  printf("--- Error: Unknown with code %d\n", bones_errors);
    }
    fflush(stdout); exit(0);
  }
}

// Use a global variable for the device ID, context and command queue
cl_device_id bones_device;
cl_context bones_context;
cl_command_queue bones_queue;

// Use a global variable to store the name and the binary for the last program
char bones_last_program[1024];
cl_program bones_program;

// Function to initialize the OpenCL platform (create to ensure fair measurements afterwards)
void bones_initialize_target(void) {
  cl_int bones_errors;
  
  // Get OpenCL platform count
  cl_uint bones_num_platforms;
  bones_errors = clGetPlatformIDs(0,NULL,&bones_num_platforms); error_check(bones_errors);
  if (bones_num_platforms == 0) { printf("Error: No OpenCL platforms found.\n"); exit(1); }
  
  // Get all OpenCL platform IDs
  cl_platform_id bones_platform_ids[10];
  bones_errors = clGetPlatformIDs(bones_num_platforms,bones_platform_ids,NULL); error_check(bones_errors);  
  
  // Select the AMD APP platform
  char bones_buffer[1024];
  cl_uint bones_platform;
  for(cl_uint bones_platform_id=0; bones_platform_id<bones_num_platforms; bones_platform_id++) {
    clGetPlatformInfo(bones_platform_ids[bones_platform_id], CL_PLATFORM_NAME, 1024, bones_buffer, NULL);
    if(strstr(bones_buffer,"Intel") != NULL) { bones_platform = bones_platform_id; break; }
  }
  
  // Get a CPU device on the platform
  bones_errors = clGetDeviceIDs(bones_platform_ids[bones_platform], CL_DEVICE_TYPE_CPU, 1, &bones_device, NULL); error_check(bones_errors);
  bones_errors = clGetDeviceInfo(bones_device, CL_DEVICE_NAME, sizeof(bones_buffer), bones_buffer, NULL); error_check(bones_errors);
  
  // Create a context
  bones_context = clCreateContext(NULL,1,&bones_device,NULL,NULL,&bones_errors); error_check(bones_errors);
  
  // Create a command queue
  bones_queue = clCreateCommandQueue(bones_context,bones_device,CL_QUEUE_PROFILING_ENABLE,&bones_errors); error_check(bones_errors);
  
  // Create space on the device
  cl_mem bones_device_data = clCreateBuffer(bones_context,CL_MEM_READ_WRITE,4,NULL,&bones_errors); error_check(bones_errors);
  
  // Copy something to the device
  bones_device_data = clCreateBuffer(bones_context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,4,bones_buffer,NULL);
  
  // Clean-up the OpenCL context
  strcpy(bones_last_program,"");
  clReleaseMemObject(bones_device_data);
  fflush(stdout);
}

// Declaration of the original function
int bones_main(void);

// New main function for initialisation and clean-up
int main(void) {
  
  // Initialisation
  bones_initialize_target();
  
  // Original main function
  int bones_return = bones_main();
  
  // Clean-up
  clReleaseCommandQueue(bones_queue);
  clReleaseProgram(bones_program);
  clReleaseContext(bones_context);
  return bones_return;
}

