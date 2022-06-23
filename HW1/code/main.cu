#include <stdio.h>
#include <stdlib.h>
#include "config.h"

//**************************************************************************
//
//  Finish your code here if need.
//
//**************************************************************************


__global__ void addmat(const float* a, const float* b, float* c, int height, int width){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i<height&&j<width){
        c[i*width+j] = a[i*width+j] + b[i*width+j];
    }

}




int main(int argc, char* argv[]) {
    if (argc == 3) {
        inputPath = argv[1];
        outputPath = argv[2];
    }

    // Open the input file
    FILE *stream = fopen(inputPath, "rb");
    if (stream == NULL) {
        printf("failed to open the data file %s\n", inputPath);
        return -1;
    }

    // Open a stream to write out results in text
    FILE *outStream = fopen(outputPath, "wb");
    if (outStream == NULL) {
        printf("failed to open the output file %s\n", outputPath);
        return -1;
    }

    // Read in and process the input matrix one-by-one
    int width, height, size;
    float *input1, *input2, *result;
    loadMatrix(stream, &width, &height, &input1);
    loadMatrix(stream, &width, &height, &input2);
    size = width * height;
    result = (float*)malloc(sizeof(float) * size);

    //**************************************************************************
    //
    //  Finish your code here.  Node that the array is 1D, so you should 
    //  visit the element of matrix with the way such as input[i*width+j].
    //
    //**************************************************************************



    float *a, *b, *c;
    cudaMalloc((void **)&a, sizeof(float)*height*width);
    cudaMalloc((void **)&b, sizeof(float)*height*width);
    cudaMalloc((void **)&c, sizeof(float)*height*width);

    cudaMemcpy(a, input1, sizeof(float)*height*width, cudaMemcpyHostToDevice);
    cudaMemcpy(b, input2, sizeof(float)*height*width, cudaMemcpyHostToDevice);

    dim3 gridDim(128, 32);
    dim3 blockDim(32, 32);
    
    clock_t start, end;
    start = clock();
    addmat<<<gridDim, blockDim>>>(a, b, c, height, width);
    end = clock();
    
    printf("GPU_Time: %f s.\n", (float)(end-start)/CLOCKS_PER_SEC);

    cudaMemcpy(c, result, sizeof(float)*height*width, cudaMemcpyDeviceToHost);

    saveMatrix(outStream, &width, &height, &result);

    // for(int i=0;i<height;i++){
    //     for(int j=0;j<width;j++){
    //         printf("%.1f ", result[i*width+j]);
    //     }
    //     printf("**%d\n", i);
    // }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    // De-allocate the nput and the result
    free(input1);
    free(input2);
    input1 = input2 = NULL;
    free(result);
    result = NULL;
    

    // Close the output stream
    fclose(outStream);
    return 0;
}