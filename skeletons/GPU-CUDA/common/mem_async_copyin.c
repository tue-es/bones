
// Copy <array> to the device
void bones_copy<direction>_<id>_<array>(<definition>) {
  cudaStreamSynchronize(kernel_stream);
  bones_memcpy(device_<array>, <array><flatten>, <variable_dimensions>*sizeof(<type>), cudaMemcpyHostToDevice, <state>, <index>);
}