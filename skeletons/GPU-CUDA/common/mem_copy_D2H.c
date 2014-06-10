  
  // Copy <array> from device to host
  bones_memcpy(<array><flatten>+<offset>, device_<array>+<offset>, <variable_dimensions>*sizeof(<type>), cudaMemcpyDeviceToHost, <state>, <state>);
  bones_synchronize(<state>);
