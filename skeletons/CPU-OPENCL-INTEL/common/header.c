#include <stdlib.h>

void bones_timer_start();
void bones_timer_stop();

// Allocate a 128-byte aligned pointer
void *bones_malloc_128(size_t bones_size) {
  char *bones_pointer;
  char *bones_pointer2;
  char *bones_aligned_pointer;
  
  // Allocate the memory plus a little bit extra
  bones_pointer = (char *)malloc(bones_size + 128 + sizeof(int));
  if(bones_pointer==NULL) { return(NULL); }
  
  // Create the aligned pointer
  bones_pointer2 = bones_pointer + sizeof(int);
  bones_aligned_pointer = bones_pointer2 + (128 - ((size_t)bones_pointer2 & 127));
  
  // Set the padding size
  bones_pointer2 = bones_aligned_pointer - sizeof(int);
  *((int *)bones_pointer2) = (int)(bones_aligned_pointer - bones_pointer);
  
  // Return the 128-byte aligned pointer
  return (bones_aligned_pointer);
}

// Free the 128-byte aligned pointer
void bones_free_128(void *bones_pointer) {
  int *bones_pointer2 = (int *)bones_pointer - 1;
  bones_pointer = (char *)bones_pointer - *bones_pointer2;
  free(bones_pointer);
}

