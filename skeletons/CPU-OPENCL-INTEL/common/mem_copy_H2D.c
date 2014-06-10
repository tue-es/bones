  
  #if ZEROCOPY == 0
    device_<array> = clCreateBuffer(bones_context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,<variable_dimensions>*sizeof(<type>),<array><flatten>, &bones_errors); error_check(bones_errors);
    clFinish(bones_queue);
  #endif