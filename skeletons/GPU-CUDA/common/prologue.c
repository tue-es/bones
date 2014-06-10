  
  // Set the cache size to maximal
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  
  // Stop execution directly if there is no work to do
  if (<parallelism> <= 0) { return; }
