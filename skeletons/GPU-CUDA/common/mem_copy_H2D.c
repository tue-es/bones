  
  // Copy <array> to the device
  cudaMemcpy(device_<array>, <array><flatten>, <variable_dimensions>*sizeof(<type>), cudaMemcpyHostToDevice);
