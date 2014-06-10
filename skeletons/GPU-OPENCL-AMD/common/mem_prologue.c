  
  // Create a device pointer for <array>
  cl_mem device_<array> = clCreateBuffer(bones_context,CL_MEM_READ_WRITE,<variable_dimensions>*sizeof(<type>),NULL,&bones_errors); error_check(bones_errors);
