  
  // Create space for <array> on the device
  cudaMalloc((void**)&device_<array>, <variable_dimensions>*sizeof(<type>));
  //cudaMemset((void*)device_<array>, 0, <variable_dimensions>*sizeof(<type>));
