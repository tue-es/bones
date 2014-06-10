  
  }
  
  // Stop the timer for the measurement of the kernel execution time
  clFinish(bones_queue);
  cl_ulong end2, start2;
  bones_errors = clWaitForEvents(1, &bones_event); error_check(bones_errors);
  bones_errors = clGetEventProfilingInfo(bones_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end2, 0); error_check(bones_errors);
  bones_errors = clGetEventProfilingInfo(bones_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start2, 0); error_check(bones_errors);
  float bones_timer2 = 0.000001 * (end2-start2);
  printf(">>>\t\t Execution time [kernel <algorithm_basename>]: %.3lf ms \n", bones_timer2);
