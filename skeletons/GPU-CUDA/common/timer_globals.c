
////////////////////////////////////////
//////////// Timers ////////////////////
////////////////////////////////////////

// Timer
cudaEvent_t bones_start1;

// Start the timer for the measurement of the whole scop
void bones_timer_start() {
  cudaDeviceSynchronize();
  cudaEventCreate(&bones_start1);
  cudaEventRecord(bones_start1,kernel_stream);
}

// End the timer for the measurement of the whole scop
void bones_timer_stop() {
  cudaDeviceSynchronize();
  cudaEvent_t bones_stop1;
  cudaEventCreate(&bones_stop1);
  cudaEventRecord(bones_stop1,kernel_stream);
  cudaEventSynchronize(bones_stop1);
  float bones_timer1 = 0;
  cudaEventElapsedTime(&bones_timer1,bones_start1,bones_stop1);
  printf(">>>\t\t Execution time [full scop]: %.3lf ms \n", bones_timer1);
}