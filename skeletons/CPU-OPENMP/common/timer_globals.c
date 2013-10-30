////////////////////////////////////////
//////////// Timers ////////////////////
////////////////////////////////////////

// Includes
#include <stdio.h>

// Timer
struct timeval bones_start_time1;

// Start the timer for the measurement of the whole scop
void bones_timer_start() {
  /*
  const int bones_flush_size = 4*1024*1024; // (16MB)
  char *bones_flush_c = (char *)malloc(bones_flush_size);
  for (int i=0; i<10; i++) {
    for (int j=0; j<bones_flush_size; j++) {
      bones_flush_c[j] = i*j;
    }
  }
  free(bones_flush_c);*/
  gettimeofday(&bones_start_time1, NULL);
}

// End the timer for the measurement of the whole scop
void bones_timer_stop() {
  #if (ITERS == 1)
    struct timeval bones_end_time1;
    gettimeofday(&bones_end_time1, NULL);
    float bones_timer1 = 0.001 * (1000000*(bones_end_time1.tv_sec-bones_start_time1.tv_sec)+bones_end_time1.tv_usec-bones_start_time1.tv_usec);
    printf(">>>\t\t Execution time [full scop]: %.3lf ms \n", bones_timer1);
  #endif
}