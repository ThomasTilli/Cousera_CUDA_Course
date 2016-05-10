#include	<wb.h>
#define NStream 1
__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
		int i=threadIdx.x+blockDim.x*blockIdx.x;
	if(i<len) out[i]=in1[i]+in2[i];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The input length is ", inputLength);
	
	int n;
	n=inputLength;
	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
	int size = n* sizeof(float);
	int sn = n/NStream;
	int segSize=sn* sizeof(float);
	
		
	dim3 DimGrid((sn -1)/256 +1 , 1 , 1);
	dim3 DimBlock(256 , 1, 1);

	float *d_A, *d_B, *d_C;
	wbTime_stop(GPU, "Allocating GPU memory.");
	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_C, size);

	cudaStream_t stream[NStream]; 
	for (int i = 0; i < NStream; ++i) 
		cudaStreamCreate(&stream[i]);
    wbTime_start(GPU, "Streams created");
 
	 wbTime_start(Compute, "Performing CUDA computation");
 	 //@@ Launch the GPU Kernel here
   
	for (int i = 0; i < NStream; ++i) {
		cudaMemcpyAsync(d_A + i * segSize, hostInput1 + i * segSize, segSize, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(d_B + i * segSize, hostInput2 + i * segSize, segSize, cudaMemcpyHostToDevice, stream[i]);
		vecAdd<<<DimGrid, DimBlock, 0, stream[i]>>>(d_A+ i * segSize, d_B+ i * segSize, d_C+ i * segSize,sn);
	
		cudaMemcpyAsync(hostOutput + i * segSize,d_C + i * segSize,  segSize, cudaMemcpyDeviceToHost, stream[i]);
	
	}
    wbTime_stop(Compute, "Performing CUDA computation");
	for (int i = 0; i < NStream; ++i) 
		cudaStreamDestroy(stream[i]);
  
	wbSolution(args, hostOutput, inputLength);
 	wbTime_start(GPU, "Freeing GPU Memory");
  	//@@ Free the GPU memory here
  	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  	wbTime_stop(GPU, "Freeing GPU Memory");

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

