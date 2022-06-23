#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "utils.h"

//**************************************************************************
//
//  Finish your code here if need.
//

constexpr int BDIM = 8;
constexpr int r = 2;

/*******************************/
//    “global version”:
__global__ void global_cal_entropy(int *in, float *out, int width, int height){
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int all = 0;
    int cnt[16] = {0};

    for(int i=-r; i<=r ; i++){
        for(int j=-r; j<=r; j++){
            if(tid_y+j>=0 && tid_y+j<height && tid_x+i>=0 && tid_x+i<width){
                all++;
                cnt[in[(tid_y+j)*width+tid_x+i]] += 1;
            }
        }
    }

    float t = 0.0f;
    for(int k=0;k<16;k++){
        if(cnt[k]==0)
            continue;
        t += -(float)cnt[k]/(float)all*(float)log2((float)cnt[k]/all);
    }

    out[tid_y*width+tid_x] = t;

}

/*******************************/
//    "global+log_table version": 
__global__ void log_cal_entropy(int *in, float *out, int width, int height, float *log_d){
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int all = 0;
    int cnt[16] = {0};

    for(int i=-r; i<=r ; i++){
        for(int j=-r; j<=r; j++){
            if(tid_y+j>=0 && tid_y+j<height && tid_x+i>=0 && tid_x+i<width){
                all++;
                cnt[in[(tid_y+j)*width+tid_x+i]] += 1;
            }
        }
    }

    float t = 0.0f;
    for(int k=0;k<16;k++){
        if(cnt[k]==0)
            continue;
        t += -(float)cnt[k]/(float)all*(log_d[cnt[k]]-log_d[all]);
    }

    out[tid_y*width+tid_x] = t;

}

/*******************************/
//    "share mem version": 
__global__ void share_cal_entropy(int *in, float *out, int width, int height){
    __shared__ int smem[BDIM+2*r][BDIM+2*r];
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int sid_x = threadIdx.x + r;
    int sid_y = threadIdx.y + r;

    smem[sid_y][sid_x] = in[tid_y*width+tid_x];
    if(tid_x==0&&tid_y==0){
        for(int i=0; i<BDIM+r*2; i++){
            for(int j=0; j<BDIM+r*2; j++){
                if(i<2||i>BDIM+r-1){
                    smem[i][j] = -1;
                }
                if(j<2||j>BDIM+r-1){
                    smem[i][j] = -1;
                }
            }
        }
    }
    __syncthreads();

    int all = 0;
    int cnt[16] = {0};
    for (int i=-r; i<=r; i++){
        for (int j=-r; j<=r; j++){
            if (smem[sid_y+j][sid_x+i] != -1){
                all++;
                cnt[smem[sid_y+j][sid_x+i]] += 1;
            }
        }
    } 

    float t = 0.0f;
    for(int k=0;k<16;k++){
        if(cnt[k]==0)
            continue;
        t += -(float)cnt[k]/(float)all*(float)log2((float)cnt[k]/all);
    }
    out[tid_y*width+tid_x] = t;    
    
}

//**************************************************************************

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
    int *input;
    float *result;
    loadMatrix(stream, &width, &height, &input);
    size = width * height;
    result = (float*)malloc(sizeof(float) * size);

    //**************************************************************************
    //
    //  Finish your code here.  Node that the array is 1D, so you should 
    //  visit the element of matrix with the way such as input[i*width+j].
    //
    //**************************************************************************

/***********************************/
    //global version::
    int *in;
    float *out;
    CHECK(cudaMalloc((void **)&in, sizeof(float)*size));
    CHECK(cudaMemcpy(in, input, sizeof(float)*size, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void **)&out, sizeof(float)*size));
    CHECK(cudaMemcpy(out, result, sizeof(float)*size, cudaMemcpyHostToDevice));

    
    dim3 block(BDIM, BDIM, 1);
    dim3 grid(divup(width, BDIM), divup(height, BDIM), 1);

    auto sta = getTime();
    global_cal_entropy << < grid, block >> > (in, out, width, height);
    auto end = getTime();
    auto time = end - sta;
    printf("GLOBAL USED TIME: %ld \n", time);
    
    CHECK(cudaMemcpy(result, out, sizeof(float)*size, cudaMemcpyDeviceToHost));

    // for(int i=0; i<size; i++){
    //     if(i%width == 0){
    //         printf("\n");
    //     }
    //     printf("%f ", result[i]);
    // }
    // printf("\n");

/***********************************/
    //global+log-table version::

    int len = 26;
    float *log_device;
    float *log_host = (float*)malloc(sizeof(float)*(len));
    log_host[0] = 0;

    for(int i=1; i<len; i++){
        log_host[i] = log2(i);
    }
    CHECK(cudaMalloc((void**)&log_device, sizeof(float)*len));
    CHECK(cudaMemcpy(log_device, log_host, sizeof(float)*len, cudaMemcpyHostToDevice));

    auto log_sta = getTime();
    log_cal_entropy << < grid, block >> > (in, out, width, height, log_device);
    auto log_end = getTime();
    auto log_time = log_end - log_sta;
    printf("GLOBAL+Log_table USED TIME: %ld \n", log_time);

    CHECK(cudaMemcpy(result, out, sizeof(float)*size, cudaMemcpyDeviceToHost));

    // for(int i=0; i<size; i++){
    //     if(i%width == 0){
    //         printf("\n");
    //     }
    //     printf("%f ", result[i]);
    // }
    // printf("\n");

/***********************************/
    //share version::
    auto share_sta = getTime();
    share_cal_entropy << < grid, block >> > (in, out, width, height);
    auto share_end = getTime();
    auto share_time = share_end - share_sta;
    printf("SHARE USED TIME: %ld \n", share_time);
    
    CHECK(cudaMemcpy(result, out, sizeof(float)*size, cudaMemcpyDeviceToHost));

    // for(int i=0; i<size; i++){
    //     if(i%width == 0){
    //         printf("\n");
    //     }
    //     printf("%f ", result[i]);
    // }
    // printf("\n");

    free(log_host);
    log_host = NULL;
    CHECK(cudaFree(in));
    CHECK(cudaFree(out));
    CHECK(cudaFree(log_device));

    //***************************************************************************
       
    saveMatrix(outStream, &width, &height, &result);

    // De-allocate the nput and the result
    free(input);
    input = NULL;
    free(result);
    result = NULL;
    
    // Close the stream
    fclose(stream);
    fclose(outStream);
    return 0;
}