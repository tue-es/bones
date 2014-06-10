
// Multiple iterations for measurements
#define ITERS 1

// Declaration of the original function
int bones_main(void);

// New main function for initialisation and clean-up
int main(void) {
  
  // Original main function
  int bones_return = bones_main();
  
  // Clean-up
  return bones_return;
}

