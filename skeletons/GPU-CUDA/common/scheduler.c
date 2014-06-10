
#include <stdio.h>
#include <pthread.h>

////////////////////////////////////////
////////// Thread scheduler ////////////
////////////////////////////////////////

// Memory copy and kernel streams
cudaStream_t kernel_stream;
cudaStream_t memory_stream;

// Task structure
typedef struct {
  void *dst;
  void *src;
  int size;
  enum cudaMemcpyKind direction;
  int deadline;
  volatile int status;
} Task;

// Task list
#define BONES_MAX_TASKS 100
Task tasks[BONES_MAX_TASKS];

// Scheduler status
volatile int bones_scheduler_done;

// Create synchronisation points
void bones_synchronize(int deadline) {
  cudaStreamSynchronize(kernel_stream);
  printf("Reached: syncpoint %d [worker]\n",deadline); fflush(stdout);
  for (int t = 0; t <= BONES_MAX_TASKS; t++) {
    if (tasks[t].deadline == deadline && tasks[t].status == 1) {
      while(tasks[t].status != 2) { }
    }
  }
  printf("Reached: syncpoint %d [all]\n",deadline); fflush(stdout);
}

// Add a new task
void bones_memcpy(void *dst, void *src, int size, enum cudaMemcpyKind direction, int deadline, int task_id) {
  Task new_task = { .dst = dst, .src = src, .size = size, .direction = direction, .deadline = deadline, .status = 1 };
  tasks[task_id] = new_task;
}

// Perform a task (CUDA memory copy)
void bones_scheduler_copy(Task current_task) {
  //usleep(400);
  cudaMemcpyAsync(current_task.dst, current_task.src, current_task.size, current_task.direction, memory_stream);
  cudaStreamSynchronize(memory_stream);
}

// Initialize the scheduler
void bones_initialize_scheduler(void) {
  bones_scheduler_done = 0;
}

// The scheduler (infinite loop)
#define LARGE_INT 1000
void* bones_scheduler(void* ptr) {
  cudaStreamCreate(&memory_stream);
  while (bones_scheduler_done != 1) {
    
    // Find the ready task with the earliest deadline
    int found_deadline = LARGE_INT;
    int found_task = LARGE_INT;
    for (int t = 0; t <= BONES_MAX_TASKS; t++) {
      if (tasks[t].status == 1) {
        if (tasks[t].deadline < found_deadline) {
          found_task = t;
          found_deadline = tasks[t].deadline;
        }
      }
    }
    
    // Perform the found task
    if (found_task != LARGE_INT) {
      //printf("Performing task %d, dl %d [scheduler]\n",found_task,tasks[found_task].deadline);
      bones_scheduler_copy(tasks[found_task]);
      tasks[found_task].status = 2;
    }
  }
  cudaStreamDestroy(memory_stream);
}
