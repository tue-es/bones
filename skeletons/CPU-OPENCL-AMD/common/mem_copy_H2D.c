  
  // Copy <array> to the device
  device_<array> = clCreateBuffer(bones_context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,<variable_dimensions>*sizeof(<type>),<array><flatten>,NULL);
  clFinish(bones_queue);
