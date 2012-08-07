/* STARTDEF
void bones_verify_results_<name>(<type> *bones_a, <type> *bones_b, <argument_definition>);
ENDDEF */
void bones_verify_results_<name>(<type> *bones_a, <type> *bones_b, <argument_definition>) {
  long bones_m=0;
  long bones_e=0;
  for (int bones_global_id=0; bones_global_id<<dimensions>; bones_global_id++) {
    <verifyids>
    int bones_id = <flatindex>;
    if (fabs(bones_a[bones_id]) > 0.000000001 ) {
      if ((fabs((bones_b[bones_id]/bones_a[bones_id])-1) < 0.001)) { bones_m++; } else { bones_e++; }
    } else {
      if (fabs(bones_a[bones_id]-bones_b[bones_id]) < 0.001) { bones_m++; } else { bones_e++; }
    }
    //printf("%.3lf versus %.3lf\n",bones_a[bones_id],bones_b[bones_id]);
    //printf("%d versus %d\n",bones_a[bones_id],bones_b[bones_id]);
  }
  printf("*** Verification ");
  if (bones_e == 0) { printf("complete: no errors found.\n"); }
  else { printf("warning: found %li (%.1lf%%) error(s).\n", bones_e, (bones_e*100.0)/(bones_e+bones_m)); }
  
}

