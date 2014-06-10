
////////////////////////////////////////
//////////// Timers ////////////////////
////////////////////////////////////////

// Timer
struct timeval bones_start_time1;

// Start the timer for the measurement of the whole scop
void bones_timer_start() {
  clFinish(bones_queue);
  gettimeofday(&bones_start_time1, NULL);
}

// End the timer for the measurement of the whole scop
void bones_timer_stop() {
  #if (ITERS == 1)
    clFinish(bones_queue);
    struct timeval bones_end_time1;
    gettimeofday(&bones_end_time1, NULL);
    float bones_timer1 = 0.001 * (1000000*(bones_end_time1.tv_sec-bones_start_time1.tv_sec)+bones_end_time1.tv_usec-bones_start_time1.tv_usec);
    printf(">>>\t\t Execution time [full scop]: %.3lf ms \n", bones_timer1);
  #endif
}