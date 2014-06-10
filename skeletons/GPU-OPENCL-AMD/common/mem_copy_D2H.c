  
  // Perform a copy of <array> from device to host
  clEnqueueReadBuffer(bones_queue,device_<array>,CL_TRUE,(<offset>)*sizeof(<type>),<variable_dimensions>*sizeof(<type>),<array><flatten>+<offset>,0,NULL,NULL);
  clFinish(bones_queue);
