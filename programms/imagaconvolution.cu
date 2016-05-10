#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define TILE_WIDTH 16
#define RADIUS  Mask_width/2
#define AREA (TILE_WIDTH +  Mask_width - 1)
#define GRID_SIZE(x) (ceil((float)x/TILE_WIDTH))

__device__ inline void setIndexes(unsigned int d,
                                  unsigned int &dX,
                                  unsigned int &dY,
                                  int &sX, int &sY){
  dX = d % AREA;
  dY = d / AREA;
  sX = blockIdx.x * TILE_WIDTH + dX - RADIUS;
  sY = blockIdx.y * TILE_WIDTH + dY - RADIUS;
}

//@@ INSERT CODE HERE
__global__ void convolution(float* I, const float* __restrict__ M, float* P,
                            int channels, int width, int height) {
  __shared__ float tmp[AREA][AREA];

  float acc;
  int sourceY, sourceX;
  unsigned int source, destination, destinationY, destinationX;
  unsigned int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
  unsigned int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

  for (unsigned int k = 0; k < channels; k++) {
    destination = threadIdx.y * TILE_WIDTH + threadIdx.x;
    setIndexes(destination,
               destinationX,
               destinationY,
               sourceX, sourceY);
    source = (sourceY * width + sourceX) * channels + k;
    if (sourceY >= 0 && sourceY < height && sourceX >= 0 && sourceX < width)
      tmp[destinationY][destinationX] = I[source];
    else
      tmp[destinationY][destinationX] = 0;

    destination = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    setIndexes(destination,
               destinationX,
               destinationY,
               sourceX, sourceY);
    source = (sourceY * width + sourceX) * channels + k;

    if (destinationY < AREA)
      if (sourceY >= 0 && sourceY < height && sourceX >= 0 && sourceX < width)
        tmp[destinationY][destinationX] = I[source];
      else
        tmp[destinationY][destinationX] = 0;

    __syncthreads();

    acc = 0;
    #pragma unroll
    for (unsigned int i = 0; i <  Mask_width; i++)
      #pragma unroll
      for (unsigned int j = 0; j <  Mask_width; j++)
        acc += tmp[threadIdx.y + i][threadIdx.x + j] * M[i *  Mask_width + j];

    if (y < height && x < width) P[(y * width + x) * channels + k] = min(max(acc, 0.0), 1.0);

    __syncthreads();
  }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	unsigned int dimGridX = GRID_SIZE(imageWidth);
	unsigned int dimGridY = GRID_SIZE(imageHeight);
	dim3 dimGrid(dimGridX, dimGridY);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	convolution<<<dimGrid, dimBlock>>>(deviceInputImageData,
									   deviceMaskData,
									   deviceOutputImageData,
									   imageChannels,
									   imageWidth,
									   imageHeight);
	wbCheck( cudaThreadSynchronize() );
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
