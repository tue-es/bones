
// Clean up array <array> from the device
void bones_free_<id>_<array>(void) {
  cudaStreamSynchronize(kernel_stream);
  cudaFree(device_<array>);
}