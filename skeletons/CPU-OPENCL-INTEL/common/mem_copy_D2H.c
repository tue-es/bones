  
  // Perform a zero-copy of <array> from device to host
  #if ZEROCOPY == 1
    printf("Copying back from device_<array> to <array>\n");
    void* bones_pointer_to_<array> = clEnqueueMapBuffer(bones_queue,device_<array>,CL_TRUE,CL_MAP_READ,0,<variable_dimensions>*sizeof(<type>),0,NULL,NULL,&bones_errors); error_check(bones_errors);
    clEnqueueUnmapMemObject(bones_queue,device_<array>,bones_pointer_to_<array>,0,NULL,NULL);
  #elif ZEROCOPY == 0
    bones_errors = clEnqueueReadBuffer(bones_queue,device_<array>,CL_TRUE,(0)*sizeof(<type>),<variable_dimensions>*sizeof(<type>),<array><flatten>+0,0,NULL,NULL); error_check(bones_errors);
  #endif
  clFinish(bones_queue);
