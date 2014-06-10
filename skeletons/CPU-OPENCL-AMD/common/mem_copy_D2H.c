  
  // Perform a zero-copy of <array> from device to host
  //void* bones_pointer_to_<array> = clEnqueueMapBuffer(bones_queue,device_<array>,CL_TRUE,CL_MAP_READ,<offset>,<variable_dimensions>*sizeof(<type>),0,NULL,NULL,&bones_errors); error_check(bones_errors);
  //clEnqueueUnmapMemObject(bones_queue,device_<array>,bones_pointer_to_<array>,0,NULL,NULL);
  
  // Perform a copy of <array> from device to host
  clEnqueueReadBuffer(bones_queue,device_<array>,CL_TRUE,(<offset>)*sizeof(<type>),<variable_dimensions>*sizeof(<type>),<array><flatten>+<offset>,0,NULL,NULL);
  clFinish(bones_queue);
