// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

//@@ insert code here


#define TILE_WIDTH 16

void ucharImgSeq(float * rgbImage, unsigned char * ucharImage,int width, int height, int channels) {
	
	for (int i=0; i< (width * height * channels);i++){ 
           ucharImage[i] = (unsigned char) (255 * rgbImage[i]);

	}
	
}


void grayImgSeq(unsigned char * ucharImage, unsigned char * grayImage,int width, int height, int channels) {
	for(int i=0;i<height;i++) {
		for(int j=0;j<width;j++) {
    
        int idx = i * width + j;
        // here channels is 3
        unsigned char r = ucharImage[3*idx];
        unsigned char g = ucharImage[3*idx + 1];
        unsigned char b = ucharImage[3*idx + 2];
        grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
	  }
    }
	
}

void floatImgSeq(float * rgbImage, unsigned char * ucharImage,int width, int height, int channels) {
	
	for (int i=0; i< (width * height * channels);i++){ 
           rgbImage[i] = (float) (ucharImage[i]/255.0);

	}
	
}



inline unsigned char clampSeq(float  x, float start, float end) {
    return (unsigned char) min(max(x, start), end);
}


inline unsigned char  correct_colorSeq(float* cdf,float cdfmin, unsigned char val) {
	float  v= 255*(cdf[val] - cdfmin)/(1 - cdfmin);
	return clampSeq(v, 0.0,255.0);
}	

void equalizeSeq(unsigned char * ucharImage,int width, int height,int channels, float *cdf,float cdfmin) {
	for(int i=0;i<width * height * channels;i++) {
		ucharImage[i]=correct_colorSeq(cdf,cdfmin,ucharImage[i]);
	}	
}	

void histogramSeq(unsigned char * grayImage,int width, int height,float* histogram) {
	for (int i=0;i< width * height;i++) {

       histogram[grayImage[i]]++;
		
	}		
}	


inline float p(float x, int width, int height) {
  return x/(1.0*width*height);
  
}

void cdf(float* hist, float* cumdf, int width, int height) {
	cumdf[0] = p(hist[0],width,height);
	for(int i=0;i<	HISTOGRAM_LENGTH;i++) {
		cumdf[i] = cumdf[i - 1] + p(hist[i],width,height);
	}	
}

float minCdf(float *cdf) {
	float x=0.0;
	for(int i=0;i<	HISTOGRAM_LENGTH;i++) {
	  x=min(x,cdf[i]);
	}	
	return x;
}	
//----------------------------------------------------------------------------
//Parallel Code

__global__  void ucharImg(float * rgbImage, unsigned char * ucharImage,int width, int height, int channels) {
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	if (i< width * height * channels){ 
           ucharImage[i] = (unsigned char) (255 * rgbImage[i]);

	}
	
}


__global__ void floatImg(float * rgbImage, unsigned char * ucharImage,int width, int height, int channels) {
	
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	if (i< width * height * channels){ 
           rgbImage[i] = (float) (ucharImage[i]/255.0);

	}
	
}



__global__ void grayImg(unsigned char * ucharImage, unsigned char * grayImage,int width, int height, int channels) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if((i<height) &&(j<width))  {
    
	    int idx = i * width + j;
		// here channels is 3
		unsigned char r = ucharImage[3*idx];
		unsigned char g = ucharImage[3*idx + 1];
		unsigned char b = ucharImage[3*idx + 2];
		grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
	 
    }
	
}




__device__ inline unsigned char clamp(float  x, float start, float end) {
    return (unsigned char) min(max(x, start), end);
	
}

__device__ inline unsigned char  correct_color(float* cdf,float cdfmin, unsigned char val) {
	float  v= 255*(cdf[val] - cdfmin)/(1 - cdfmin);
	return clamp(v, 0.0,255.0);
}	
	

__global__ void equalize(unsigned char * ucharImage,int width, int height,int channels, float *cdf,float cdfmin) {
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	if (i< width * height * channels){ 
		ucharImage[i]=correct_color(cdf,cdfmin,ucharImage[i]);
	//	ucharImage[i]=clamp(1.0*ucharImage[i],0.0,255.0);
	}	
}	

__global__ void histogram(unsigned char * grayImage,int width, int height,float* histogram) {
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	if ( i< width * height) {

       atomicAdd(&(histogram[grayImage[i]]),1);
	}		
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
	
	float * deviceInputImageData;
    float * deviceOutputImageData;
	unsigned char * devucharImage;
	unsigned char * devgrayImage;
	float  * devCumdf;
		unsigned char * ucharImage;
	float * hist;
	float * cumdf;
	float cdfmin;
	
	float * devHist;
	

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	
	int size=imageWidth*imageHeight;
	int imSize=size*imageChannels;
	int imFSize=imSize*sizeof(float);
	int imUSize=imSize  * sizeof(unsigned char);
	int imGraySize=size  * sizeof(unsigned char);
	
	ucharImage= (unsigned char * )malloc(imUSize);
	hist=(float *) malloc(HISTOGRAM_LENGTH *sizeof(float));
	cumdf=(float *) malloc(HISTOGRAM_LENGTH *sizeof(float));
	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    
	
	wbTime_stop(Generic, "Importing data and creating memory on host");
	
	wbTime_start(GPU, "Allocating GPU memory.");
	//Input and Output Data GPU
	wbCheck(cudaMalloc((void **) &deviceInputImageData, imFSize));
	wbCheck(cudaMalloc((void **) &deviceOutputImageData, imFSize));
	
	wbCheck(cudaMalloc((void **) &devucharImage, imUSize));
	wbCheck(cudaMalloc((void **) &devgrayImage, imGraySize));
	wbCheck(cudaMalloc((void **) &devCumdf, HISTOGRAM_LENGTH *sizeof(float)));
	wbCheck(cudaMalloc((void **) &devHist, HISTOGRAM_LENGTH *sizeof(float)));
	
	 wbTime_stop(GPU, "Allocating GPU memory.");
   
	 wbTime_start(GPU, "Copying input memory to the GPU.");
  
    wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imFSize, cudaMemcpyHostToDevice));
  	wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ insert code here
    // Initialize the grid and block dimensions here
     dim3 gridUchar((imSize-1)/256 + 1, 1, 1);
     dim3 blockUchar(256, 1, 1);
	
  //  ucharImgSeq(hostInputImageData,ucharImage,imageWidth,imageHeight, imageChannels);	
//	grayImgSeq(ucharImage,grayImage,imageWidth,imageHeight, imageChannels);
	 wbTime_start(GPU, "Transforming Image and histogramm calulation.");
	ucharImg<<<gridUchar,blockUchar>>>(deviceInputImageData,devucharImage,imageWidth,imageHeight, imageChannels);
	cudaMemcpy(ucharImage, devucharImage, imUSize, cudaMemcpyDeviceToHost);

	dim3 dimGrid;
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dimGrid.x = (imageWidth + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (imageHeight + dimBlock.y - 1)/dimBlock.y;
	grayImg<<<dimGrid,dimBlock>>>(devucharImage,devgrayImage,imageWidth,imageHeight, imageChannels);
	
	histogram<<<gridUchar,blockUchar>>>(devgrayImage,imageWidth,imageHeight,devHist);
	
    cudaMemcpy(hist, devHist,  HISTOGRAM_LENGTH *sizeof(float), cudaMemcpyDeviceToHost);	
	wbTime_stop(GPU, "Transforming Image and histogramm calulation.");
	cdf(hist, cumdf,imageWidth,imageHeight);
	cdfmin=minCdf(cumdf);		
	
			
	wbCheck(cudaMemcpy(devCumdf, cumdf, HISTOGRAM_LENGTH *sizeof(float), cudaMemcpyHostToDevice));		
	 wbTime_start(GPU, "Equalizing Image .");
   equalize<<<gridUchar,blockUchar>>>(devucharImage,imageWidth,imageHeight,imageChannels, devCumdf,cdfmin);

  

	floatImg<<<gridUchar,blockUchar>>>(deviceOutputImageData,devucharImage,imageWidth,imageHeight, imageChannels);

	wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imFSize, cudaMemcpyDeviceToHost));
	wbTime_stop(GPU, "Equalizing Image .");
	//free memory
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(devucharImage);
	cudaFree(devgrayImage);
	cudaFree(devCumdf);
    wbSolution(args, outputImage);

    //@@ insert code here

    return 0;
}

