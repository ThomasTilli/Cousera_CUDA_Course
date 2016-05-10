// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void add(float *auxiliary, float *output, int len) {
        unsigned int index = threadIdx.x;
    unsigned int start = 2 * blockDim.x * blockIdx.x;
 
    if (blockIdx.x > 0 && (start + 2 * index) < len) {
        output[start + 2 * index] += auxiliary[blockIdx.x - 1];
    }
 
        if (blockIdx.x > 0 && (start + 2 * index + 1) < len) {
        output[start + 2 * index + 1] += auxiliary[blockIdx.x - 1];
    }
}


__global__ void scan(float * input, float * output, int len,float *auxiliary) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	__shared__ float temp[2 * BLOCK_SIZE];
 
        unsigned int index = threadIdx.x;
    unsigned int start = 2 * blockDim.x * blockIdx.x;
 
 
    if ((start + 2 * index) < len) {
        temp[2 * index] = input[start + 2 * index];
    } else {
        temp[2 * index] = 0.0;
    }
 
    if ((start + 2 * index + 1) < len) {
        temp[2 * index + 1] = input[start + 2 * index + 1];
    } else {
        temp[2 * index + 1] = 0.0;
    }
 
    int offset = 1;
 
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
 
        if (index < d) {
            unsigned int ai = 2 * offset * index + offset - 1;
            unsigned int bi = 2 * offset * index + 2 * offset - 1;
            temp[bi] += temp[ai];
        }
 
        offset *= 2;
    }
 
    if (index == 0) { temp[2 * blockDim.x - 1] = 0; }
 
    for (int d = 1; d <= blockDim.x; d <<= 1) {
        offset >>= 1;
 
        __syncthreads();
 
        if (index < d) {
            unsigned int ai = 2 * offset * index + offset - 1;
            unsigned int bi = 2 * offset * index + 2 * offset - 1;
 
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
 
    __syncthreads();
 
        if ((start + 2 * index) < len) {
        output[start + 2 * index] = temp[2 * index] + input[start + 2 * index];
        }
 
        if ((start + 2 * index + 1) < len) {
                output[start + 2 * index + 1] = temp[2 * index + 1] + input[start
+ 2 * index + 1];
    }
 
        if (len > 2 * BLOCK_SIZE && index == blockDim.x - 1) {
        if ((start + 2 * blockDim.x - 1) < len) {
                        auxiliary[blockIdx.x] = output[start + 2 * blockDim.x - 1];
        } else {
                auxiliary[blockIdx.x] = output[len - 1];
        }
        }
}


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil((float) numElements / (2 * BLOCK_SIZE)), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
 

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    float *hostAuxiliary;
    float *deviceAuxiliary;
 
    if (numElements > 2 * BLOCK_SIZE) {
        wbLog(TRACE, "initialize auxiliary ", numElements);
        hostAuxiliary = (float *) malloc(ceil((float) numElements / (2* BLOCK_SIZE)) * sizeof(float));
        wbCheck(cudaMalloc((void **) &deviceAuxiliary, ceil((float)numElements / (2 * BLOCK_SIZE)) * sizeof(float)));
        wbCheck(cudaMemset(deviceAuxiliary, 0, ceil((float) numElements /(2 * BLOCK_SIZE)) * sizeof(float)));
    }
 
    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput,numElements, deviceAuxiliary);
    cudaDeviceSynchronize();
	wbLog(TRACE, "scan executed ", numElements);
 
    if (numElements > 2 * BLOCK_SIZE) {
        wbLog(TRACE, "compute auxiliary ", numElements);
 
        wbCheck(cudaMemcpy(hostAuxiliary, deviceAuxiliary, ceil((float)numElements / (2 * BLOCK_SIZE)) * sizeof(float),
cudaMemcpyDeviceToHost));
 
        for (int i = 1; i < ceil((float) numElements / (2 * BLOCK_SIZE)); i++) {
                hostAuxiliary[i] += hostAuxiliary[i - 1];
        }
 
        wbCheck(cudaMemcpy(deviceAuxiliary, hostAuxiliary, ceil((float)numElements / (2 * BLOCK_SIZE)) * sizeof(float),
cudaMemcpyHostToDevice));
 
        add<<<dimGrid, dimBlock>>>(deviceAuxiliary, deviceOutput, numElements);
 
        cudaDeviceSynchronize();
 
        cudaFree(deviceAuxiliary);
 
        free(hostAuxiliary);
    }
 

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

