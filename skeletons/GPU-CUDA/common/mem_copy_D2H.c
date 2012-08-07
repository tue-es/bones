  
  // Copy <array> from device to host
  cudaMemcpy(<array><flatten>+<offset>, device_<array>+<offset>, <variable_dimensions>*sizeof(<type>), cudaMemcpyDeviceToHost);
