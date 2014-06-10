  
  // Create a device pointer for <array>
  #if ZEROCOPY == 1
    cl_mem device_<array> = clCreateBuffer(bones_context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, <variable_dimensions>*sizeof(<type>), <array><flatten>, &bones_errors); error_check(bones_errors);
  #elif ZEROCOPY == 0
    cl_mem device_<array> = clCreateBuffer(bones_context, CL_MEM_READ_WRITE, <variable_dimensions>*sizeof(<type>), NULL, &bones_errors); error_check(bones_errors);
  #endif
