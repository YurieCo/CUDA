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
#define O_TILE_WIDTH 14
#define TILE_WIDTH (O_TILE_WIDTH + Mask_radius)
#define Grid_Size(x) ((x) - 1) / O_TILE_WIDTH + 1

//@@ INSERT CODE HERE
__global__ void Convolution_Shared(float *N, const float * __restrict__ M, float *O, int height, int width, int channels) {
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH][3];
	int tx = threadIdx.x;
	int ty = threadIdx.y;	
	int row = ty + blockIdx.y * blockDim.y;
	int col = tx + blockIdx.x * blockDim.x;
	int row_i = row - Mask_radius;
	int col_i = col - Mask_radius;
	if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
		for(int c = 0 ; c < channels; c++)
			ds_N[ty][tx][c] = N[(row_i * width + col_i) * channels + c];
	else
		for(int c = 0 ; c < channels; c++)
			ds_N[ty][tx][c] = 0.0;
	__syncthreads();
	
	for(int c = 0; c < channels; c++)
	{
		float value = 0.0;
		for(int y = 0; y < Mask_width; y++)
		for(int x = 0; x < Mask_width; x++)
			value += ds_N[ty + y][tx + x][c] * M[y * width + x];
		O[(row * width + col) * channels + c] = value;	
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
	dim3 DimGrid(Grid_Size(imageHeight), Grid_Size(imageWidth), 1);
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	Convolution_Shared<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageHeight, imageWidth ,imageChannels);
	
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
