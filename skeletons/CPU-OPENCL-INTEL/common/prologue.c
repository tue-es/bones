  fflush(stdout);
  cl_int bones_errors;
  cl_event bones_event;
  
  // Only compile if this program is different from the last one
  if (strcmp(bones_last_program,"<algorithm_filename>") != 0) {
    strcpy(bones_last_program,"<algorithm_filename>");
    
    // Load and compile the kernel
    char *bones_source = get_source("<algorithm_filename>_device.cl");
    bones_program = clCreateProgramWithSource(bones_context,1,(const char **)&bones_source,NULL,&bones_errors); error_check(bones_errors);
    bones_errors = clBuildProgram(bones_program,0,NULL,"",NULL,NULL);
    
    // Get and print the compiler log
    char* bones_log;
    size_t bones_log_size;
    clGetProgramBuildInfo(bones_program,bones_device,CL_PROGRAM_BUILD_LOG,0,NULL,&bones_log_size);
    bones_log = (char*)malloc((bones_log_size+1)*sizeof(char));
    clGetProgramBuildInfo(bones_program,bones_device,CL_PROGRAM_BUILD_LOG,bones_log_size,bones_log, NULL);
    bones_log[bones_log_size] = '\0';
    if (strcmp(bones_log,"\n") != 0 && strcmp(bones_log,"") != 0) { printf("--------- \n--- Compilation log:\n--------- \n%s\n",bones_log); }
    free(bones_log);
    error_check(bones_errors);
  }