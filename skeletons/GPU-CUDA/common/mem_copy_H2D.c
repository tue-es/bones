  
  // Copy <array> to the device
  bones_memcpy(device_<array>, <array><flatten>, <variable_dimensions>*sizeof(<type>), cudaMemcpyHostToDevice, <state>, <state>);
  bones_synchronize(<state>);
