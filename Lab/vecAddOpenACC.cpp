#include <wb.h> 

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
 
  wbLog(TRACE, "The input length is ", inputLength);
  int n= inputLength;
  #pragma acc kernels copyin(hostInput1[0:n],hostInput2[0:n]), copyout(hostOutput[0:n])
  for(int i=0; i<n; i++) {
        hostOutput[i] = hostInput1[i] + hostInput2[i];
    }
  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
