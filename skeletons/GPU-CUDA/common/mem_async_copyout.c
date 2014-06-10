
// Copy <array> from device to host
void bones_copy<direction>_<id>_<array>(<definition>) {
  cudaStreamSynchronize(kernel_stream);
  bones_memcpy(<array><flatten>+<offset>, device_<array>+<offset>, <variable_dimensions>*sizeof(<type>), cudaMemcpyDeviceToHost, <state>, <index>);
}